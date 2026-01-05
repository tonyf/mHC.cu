# CUTLASS Python DSL Migration Plan

## Overview

This document outlines a comprehensive plan for migrating the MHC (Multi-Head Communication) CUDA kernels to the NVIDIA CUTLASS Python DSL. The migration will enable:

1. **Python-native kernel development** - Write and iterate on kernels in Python
2. **JIT compilation** - Dynamic compilation with automatic caching
3. **PyTorch integration** - Seamless tensor interoperability via DLPack
4. **Maintainability** - Single codebase vs separate CUDA/Python layers

---

## 1. Kernel Inventory

### 1.1 Core Kernels to Migrate

| Kernel | File | Complexity | Priority | Dependencies |
|--------|------|------------|----------|--------------|
| `rmsnorm_kernel` | `rmsnorm.cuh` | Medium | High | Warp reductions |
| `rmsnorm_kernel_vectorized` | `rmsnorm.cuh` | Medium | High | bf16 vectorization |
| `rmsnorm_backward_kernel` | `rmsnorm.cuh` | Medium | High | Atomic ops |
| `compute_rms_kernel` | `fused_rmsnorm_matmul.cuh` | Medium | High | Warp reductions |
| `divide_by_rms_kernel` | `fused_rmsnorm_matmul.cuh` | Low | High | Element-wise |
| `sinkhorn_knopp_kernel` | `sinkhorn_knopp.cuh` | High | High | Shared memory, tiles |
| `sinkhorn_knopp_backward_kernel` | `sinkhorn_knopp.cuh` | High | Medium | Checkpointing |
| `stream_aggregate_bf16_*` | `stream_ops.cuh` | Medium | High | bf16 conversion |
| `stream_distribute_mix_*` | `stream_ops.cuh` | Medium | High | Shared memory |
| `stream_*_backward` | `stream_ops.cuh` | High | Medium | Multi-output |

### 1.2 Utility Functions

| Function | File | Notes |
|----------|------|-------|
| `fast_exp` | `utils.cuh` | Device intrinsic `__expf` |
| `fast_sigmoid` | `utils.cuh` | 1/(1+exp(-x)) using fast reciprocal |
| `float_to_bf16` | `utils.cuh` | Type conversion kernel |
| `bf16_to_float` | `utils.cuh` | Type conversion kernel |
| `fused_h_activations` | `utils.cuh` | Combined activation computation |

### 1.3 cuBLAS Dependencies

The following operations use cuBLASLt and should remain as library calls:
- `matmul_forward` in `FusedRMSNormMatmul`
- Matrix multiplications in `FusedRMSNormMatmulBackward`
- `StreamMixTC::forward` for large expansion rates (n >= 32)

**Strategy**: Wrap cuBLAS calls in Python using existing PyTorch/cuBLAS bindings rather than reimplementing in CuTe DSL.

---

## 2. Design Principles

### 2.1 Code Organization

```
src/python/mhc/
├── __init__.py
├── layer.py                    # High-level MHC layer (existing)
├── ops.py                      # PyTorch autograd functions (existing)
├── cute_dsl/
│   ├── __init__.py
│   ├── common.py               # Shared utilities, types, constants
│   ├── reductions.py           # Warp/block reduction primitives
│   ├── rmsnorm.py              # RMSNorm forward/backward kernels
│   ├── sinkhorn.py             # Sinkhorn-Knopp kernels
│   ├── stream_ops.py           # Stream aggregation/distribution
│   ├── activations.py          # Sigmoid, exp, activation fusions
│   └── mhc_layer.py            # Complete MHC layer composition
```

### 2.2 Naming Conventions

- **Kernel functions**: `@cute.kernel` decorated, suffix `_kernel`
- **JIT host functions**: `@cute.jit` decorated, descriptive names
- **Helper functions**: Plain Python, used at compile time
- **Constants**: `BLOCK_SIZE`, `MAX_N`, `VEC_SIZE` as module-level `Constexpr`

### 2.3 Type System

```python
import cutlass
from cutlass import cute

# Standard type aliases
Float32 = cutlass.Float32
BFloat16 = cutlass.BFloat16
Int32 = cutlass.Int32

# Common block sizes as compile-time constants
BLOCK_SIZE_256 = cutlass.Constexpr[int](256)
BLOCK_SIZE_512 = cutlass.Constexpr[int](512)
WARP_SIZE = cutlass.Constexpr[int](32)
```

### 2.4 Memory Management Patterns

**Shared Memory Allocation**:
```python
@cute.kernel
def my_kernel(out, inp, N: cutlass.Int32):
    # Declare shared memory with static size
    smem = cute.shared_memory((MAX_DIM, MAX_DIM), Float32)
    
    # Or dynamic shared memory via launch parameter
    # smem = cute.dynamic_shared_memory(Float32)
```

**Vectorized Memory Access**:
```python
# Use CuTe layouts for vectorized loads
# Layout: (elements_per_thread, vec_size):(vec_size, 1)
vec_layout = cute.make_layout((num_elements, 4), (4, 1))
```

### 2.5 Reduction Patterns

**Warp Reduction (Reusable Component)**:
```python
@cute.jit
def warp_reduce_sum(val: Float32) -> Float32:
    """Butterfly reduction within a warp."""
    for offset in cutlass.range_constexpr(5):  # log2(32) = 5
        mask = 1 << (4 - offset)
        val = val + cute.shfl_xor(val, mask)
    return val

@cute.jit  
def block_reduce_sum(val: Float32, shared: cute.Tensor, 
                     thread_idx: Int32, num_warps: cutlass.Constexpr) -> Float32:
    """Two-level reduction: warp then block."""
    warp_id = thread_idx // WARP_SIZE
    lane_id = thread_idx % WARP_SIZE
    
    # Warp-level reduction
    warp_sum = warp_reduce_sum(val)
    
    # Store warp results to shared memory
    if lane_id == 0:
        shared[warp_id] = warp_sum
    cute.syncthreads()
    
    # Final reduction in first warp
    if warp_id == 0:
        val = shared[lane_id] if lane_id < num_warps else 0.0
        return warp_reduce_sum(val)
    return 0.0
```

---

## 3. Kernel Migration Details

### 3.1 RMSNorm Forward

**Current CUDA Pattern**:
```cpp
template<int BLOCK_SIZE, bool OUTPUT_RMS>
__global__ void rmsnorm_kernel(floatX* out, float* rms_out, 
                               const floatX* inp, const floatX* weight,
                               int N, int C, float eps)
```

**CuTe DSL Implementation**:
```python
@cute.kernel
def rmsnorm_forward_kernel(
    out: cute.Tensor,              # [N, C] bf16 output
    rms_out: cute.Tensor,          # [N] float32 rms values (optional)
    inp: cute.Tensor,              # [N, C] bf16 input  
    weight: cute.Tensor,           # [C] bf16 weight
    N: cutlass.Int32,
    C: cutlass.Int32,
    eps: cutlass.Float32,
    output_rms: cutlass.Constexpr  # Compile-time flag
):
    BLOCK_SIZE = 512
    
    # Thread/block indices
    idx = cute.blockIdx.x
    tid = cute.threadIdx.x
    
    if idx >= N:
        return
    
    # Shared memory for warp reduction
    num_warps = BLOCK_SIZE // 32
    s_sum_sq = cute.shared_memory((num_warps,), Float32)
    
    # Get row pointers
    x = inp[idx, :]
    o = out[idx, :]
    
    # Phase 1: Compute sum of squares
    thread_sum_sq = Float32(0.0)
    for i in range(tid, C, BLOCK_SIZE):
        val = cute.cast(Float32, x[i])
        thread_sum_sq = thread_sum_sq + val * val
    
    # Block-level reduction
    block_sum = block_reduce_sum(thread_sum_sq, s_sum_sq, tid, num_warps)
    
    # Compute RMS inverse
    if tid == 0:
        rms = cute.sqrt(block_sum / cute.cast(Float32, C) + eps)
        rms_inv = 1.0 / rms
        s_sum_sq[0] = rms_inv
        if cutlass.const_expr(output_rms):
            rms_out[idx] = rms
    cute.syncthreads()
    
    rms_inv = s_sum_sq[0]
    
    # Phase 2: Normalize and scale
    for i in range(tid, C, BLOCK_SIZE):
        val = cute.cast(Float32, x[i])
        w = cute.cast(Float32, weight[i])
        o[i] = cute.cast(BFloat16, val * rms_inv * w)
```

### 3.2 Sinkhorn-Knopp Forward

**Key Patterns to Preserve**:
- Single-block execution for small matrices (≤64x64)
- Shared memory tiling
- Iterative row/column normalization
- Fast reciprocal using `__frcp_rn`

**CuTe DSL Implementation Outline**:
```python
@cute.kernel
def sinkhorn_knopp_kernel(
    out: cute.Tensor,       # [M, N] output
    inp: cute.Tensor,       # [M, N] input
    M: cutlass.Int32,
    N: cutlass.Int32,
    num_iters: cutlass.Int32,
    eps: cutlass.Float32
):
    # Use CuTe layouts for efficient tiling
    MAX_DIM = 64
    BLOCK_SIZE = 256
    
    # Allocate shared memory
    tile = cute.shared_memory((MAX_DIM, MAX_DIM), Float32)
    row_sums = cute.shared_memory((MAX_DIM,), Float32)
    col_sums = cute.shared_memory((MAX_DIM,), Float32)
    
    tid = cute.threadIdx.x
    total_elems = M * N
    
    # Load input to shared memory
    for i in range(tid, total_elems, BLOCK_SIZE):
        r, c = i // N, i % N
        tile[r, c] = inp[r, c]
    cute.syncthreads()
    
    # Iterative normalization
    for iter in range(num_iters):
        # Row normalization
        for r in range(tid, M, BLOCK_SIZE):
            sum_val = Float32(0.0)
            for c in cutlass.range_constexpr(MAX_DIM):
                if c < N:
                    sum_val = sum_val + tile[r, c]
            row_sums[r] = sum_val
        cute.syncthreads()
        
        for i in range(tid, total_elems, BLOCK_SIZE):
            r = i // N
            row_sum = row_sums[r]
            if row_sum > eps:
                tile[i // N, i % N] = tile[i // N, i % N] / row_sum
        cute.syncthreads()
        
        # Column normalization (similar pattern)
        # ...
    
    # Store results
    for i in range(tid, total_elems, BLOCK_SIZE):
        out[i // N, i % N] = tile[i // N, i % N]
```

### 3.3 Stream Operations

**Stream Aggregate Pattern**:
```python
@cute.kernel  
def stream_aggregate_kernel(
    out: cute.Tensor,           # [B, C] output
    H_pre_activated: cute.Tensor,  # [n] activated weights (output)
    inp: cute.Tensor,           # [B, n, C] input streams
    H_pre_raw: cute.Tensor,     # [n] raw weights
    B: cutlass.Int32,
    n: cutlass.Constexpr,       # Compile-time for unrolling
    C: cutlass.Int32
):
    MAX_N = n  # Compile-time constant
    
    # Load H_pre to shared memory with sigmoid activation
    s_H_pre = cute.shared_memory((MAX_N,), Float32)
    tid = cute.threadIdx.x
    
    if tid < n:
        activated = fast_sigmoid(H_pre_raw[tid])
        s_H_pre[tid] = activated
        H_pre_activated[tid] = activated
    cute.syncthreads()
    
    # Compute weighted sum
    idx = cute.blockIdx.x * BLOCK_SIZE + tid
    if idx >= B * C:
        return
        
    b = idx // C
    c = idx % C
    
    sum_val = Float32(0.0)
    for i in cutlass.range_constexpr(MAX_N):
        sum_val = sum_val + s_H_pre[i] * inp[b, i, c]
    
    out[b, c] = cute.cast(BFloat16, sum_val)
```

---

## 4. Reusable Components

### 4.1 Core Primitives Library

Create `cute_dsl/primitives.py`:

```python
"""Reusable CUDA primitives for CuTe DSL kernels."""

import cutlass
from cutlass import cute

# ============ Constants ============
WARP_SIZE = cutlass.Constexpr[int](32)

# ============ Device Functions ============

@cute.jit
def fast_exp(x: cutlass.Float32) -> cutlass.Float32:
    """Fast exponential using CUDA intrinsic."""
    return cute.exp(x)  # Maps to __expf

@cute.jit
def fast_sigmoid(x: cutlass.Float32) -> cutlass.Float32:
    """Fast sigmoid: 1/(1+exp(-x))."""
    return cute.frcp(1.0 + fast_exp(-x))

@cute.jit
def fast_reciprocal(x: cutlass.Float32) -> cutlass.Float32:
    """Fast reciprocal using CUDA intrinsic."""
    return cute.frcp(x)  # Maps to __frcp_rn

# ============ Reduction Primitives ============

@cute.jit
def warp_reduce_sum(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly warp reduction for sum."""
    for i in cutlass.range_constexpr(5):
        offset = 16 >> i
        val = val + cute.shfl_xor(val, offset)
    return val

@cute.jit
def warp_reduce_max(val: cutlass.Float32) -> cutlass.Float32:
    """Butterfly warp reduction for max."""
    for i in cutlass.range_constexpr(5):
        offset = 16 >> i
        other = cute.shfl_xor(val, offset)
        val = cute.max(val, other)
    return val

# ============ Memory Utilities ============

@cute.jit
def load_vectorized_bf16_to_f32(
    src: cute.Tensor,  # bf16 tensor
    dst_ptr,           # f32 destination  
    idx: cutlass.Int32,
    vec_size: cutlass.Constexpr
) -> None:
    """Load bf16 elements and convert to f32."""
    for i in cutlass.range_constexpr(vec_size):
        dst_ptr[i] = cute.cast(cutlass.Float32, src[idx + i])
```

### 4.2 Block Reduction Module

Create `cute_dsl/reductions.py`:

```python
"""Block-level reduction operations."""

import cutlass
from cutlass import cute
from .primitives import warp_reduce_sum, WARP_SIZE

@cute.jit
def block_reduce_sum_2level(
    val: cutlass.Float32,
    smem: cute.Tensor,
    tid: cutlass.Int32,
    block_size: cutlass.Constexpr
) -> cutlass.Float32:
    """
    Two-level block reduction:
    1. Intra-warp reduction using shuffle
    2. Inter-warp reduction via shared memory
    
    Returns the reduced sum (valid only in thread 0).
    """
    num_warps = block_size // WARP_SIZE
    warp_id = tid // WARP_SIZE
    lane_id = tid % WARP_SIZE
    
    # Level 1: Warp reduction
    warp_sum = warp_reduce_sum(val)
    
    # Write warp results to shared memory
    if lane_id == 0:
        smem[warp_id] = warp_sum
    cute.syncthreads()
    
    # Level 2: Final reduction in first warp
    if warp_id == 0:
        val = smem[lane_id] if lane_id < num_warps else 0.0
        return warp_reduce_sum(val)
    return 0.0

@cute.jit
def block_reduce_sum_broadcast(
    val: cutlass.Float32,
    smem: cute.Tensor,
    tid: cutlass.Int32,
    block_size: cutlass.Constexpr
) -> cutlass.Float32:
    """
    Block reduction with result broadcast to all threads.
    Uses smem[0] to store and broadcast the final result.
    """
    result = block_reduce_sum_2level(val, smem, tid, block_size)
    
    if tid == 0:
        smem[0] = result
    cute.syncthreads()
    
    return smem[0]
```

### 4.3 Type Conversion Module

Create `cute_dsl/conversions.py`:

```python
"""Data type conversion utilities."""

import cutlass
from cutlass import cute

@cute.kernel
def bf16_to_f32_kernel(
    out: cute.Tensor,   # Float32 output
    inp: cute.Tensor,   # BFloat16 input
    size: cutlass.Int32
):
    """Element-wise bf16 to f32 conversion."""
    BLOCK_SIZE = 256
    idx = cute.blockIdx.x * BLOCK_SIZE + cute.threadIdx.x
    if idx < size:
        out[idx] = cute.cast(cutlass.Float32, inp[idx])

@cute.kernel
def f32_to_bf16_kernel(
    out: cute.Tensor,   # BFloat16 output
    inp: cute.Tensor,   # Float32 input
    size: cutlass.Int32
):
    """Element-wise f32 to bf16 conversion."""
    BLOCK_SIZE = 256
    idx = cute.blockIdx.x * BLOCK_SIZE + cute.threadIdx.x
    if idx < size:
        out[idx] = cute.cast(cutlass.BFloat16, inp[idx])

# Host-side launchers
@cute.jit
def bf16_to_f32(out: cute.Tensor, inp: cute.Tensor, size: int):
    """Launch bf16->f32 conversion kernel."""
    BLOCK_SIZE = 256
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    bf16_to_f32_kernel[grid, BLOCK_SIZE](out, inp, size)

@cute.jit  
def f32_to_bf16(out: cute.Tensor, inp: cute.Tensor, size: int):
    """Launch f32->bf16 conversion kernel."""
    BLOCK_SIZE = 256
    grid = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    f32_to_bf16_kernel[grid, BLOCK_SIZE](out, inp, size)
```

---

## 5. PyTorch Integration Strategy

### 5.1 Autograd Function Pattern

```python
"""PyTorch integration for CuTe DSL kernels."""

import torch
from torch.autograd import Function
from cutlass.cute.runtime import from_dlpack

# Import DSL kernels
from .cute_dsl import rmsnorm, sinkhorn, stream_ops

class RMSNormDSL(Function):
    @staticmethod
    def forward(ctx, inp, weight, eps):
        # Convert PyTorch tensors to CuTe tensors
        inp_cute = from_dlpack(inp).mark_layout_dynamic()
        weight_cute = from_dlpack(weight).mark_layout_dynamic()
        
        # Allocate outputs
        out = torch.empty_like(inp)
        rms = torch.empty(inp.size(0), dtype=torch.float32, device=inp.device)
        
        out_cute = from_dlpack(out).mark_layout_dynamic()
        rms_cute = from_dlpack(rms).mark_layout_dynamic()
        
        # Launch kernel
        B, C = inp.shape
        rmsnorm.rmsnorm_forward(out_cute, rms_cute, inp_cute, weight_cute, 
                                B, C, eps, output_rms=True)
        
        # Save for backward
        ctx.save_for_backward(inp, weight, rms)
        ctx.eps = eps
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, rms = ctx.saved_tensors
        
        # Convert tensors
        grad_cute = from_dlpack(grad_output).mark_layout_dynamic()
        inp_cute = from_dlpack(inp).mark_layout_dynamic()
        weight_cute = from_dlpack(weight).mark_layout_dynamic()
        rms_cute = from_dlpack(rms).mark_layout_dynamic()
        
        # Allocate gradient outputs
        d_inp = torch.empty_like(inp, dtype=torch.float32)
        d_weight = torch.zeros(weight.size(0), dtype=torch.float32, 
                               device=weight.device)
        
        d_inp_cute = from_dlpack(d_inp).mark_layout_dynamic()
        d_weight_cute = from_dlpack(d_weight).mark_layout_dynamic()
        
        # Launch backward kernel
        B, C = inp.shape
        rmsnorm.rmsnorm_backward(d_inp_cute, d_weight_cute, grad_cute,
                                 inp_cute, weight_cute, rms_cute, B, C)
        
        return d_inp.to(inp.dtype), d_weight, None


def rmsnorm_dsl(inp, weight, eps=1e-5):
    """Functional interface for RMSNorm using CuTe DSL."""
    return RMSNormDSL.apply(inp, weight, eps)
```

### 5.2 Tensor Caching for Performance

```python
class TensorCache:
    """Cache CuTe tensor wrappers to avoid repeated from_dlpack calls."""
    
    def __init__(self):
        self._cache = {}
    
    def get_or_create(self, tensor: torch.Tensor, key: str):
        cache_key = (key, tensor.data_ptr(), tensor.shape, tensor.stride())
        if cache_key not in self._cache:
            self._cache[cache_key] = from_dlpack(tensor).mark_layout_dynamic()
        return self._cache[cache_key]
    
    def clear(self):
        self._cache.clear()

# Global cache instance
_tensor_cache = TensorCache()
```

---

## 6. Migration Task List

### Phase 1: Infrastructure (Week 1)
- [ ] Set up `cute_dsl/` module structure
- [ ] Implement `primitives.py` with fast_exp, fast_sigmoid, warp reductions
- [ ] Implement `reductions.py` with block-level reductions
- [ ] Implement `conversions.py` with bf16/f32 conversions
- [ ] Create test infrastructure with PyTorch reference implementations

### Phase 2: RMSNorm (Week 2)
- [ ] Implement `rmsnorm_forward_kernel` (non-vectorized)
- [ ] Implement `rmsnorm_forward_kernel_vectorized`
- [ ] Implement `rmsnorm_backward_kernel`
- [ ] Add PyTorch autograd wrapper
- [ ] Validate against existing CUDA implementation
- [ ] Benchmark and optimize

### Phase 3: Sinkhorn-Knopp (Week 3)
- [ ] Implement `sinkhorn_knopp_single_block_kernel` (M,N ≤ 64)
- [ ] Implement `sinkhorn_knopp_warp_optimized_kernel` (32x32 special case)
- [ ] Implement `sinkhorn_knopp_batched_kernel`
- [ ] Implement `sinkhorn_knopp_backward_kernel`
- [ ] Add fused exp variant
- [ ] Validate and benchmark

### Phase 4: Stream Operations (Week 4)
- [ ] Implement `stream_aggregate_kernel` (basic)
- [ ] Implement `stream_aggregate_vectorized_kernel`
- [ ] Implement `stream_distribute_mix_add_kernel`
- [ ] Implement backward kernels
- [ ] Add dynamic H variants
- [ ] Validate and benchmark

### Phase 5: Integration (Week 5)
- [ ] Create unified `MHCLayerDSL` class
- [ ] Integrate with cuBLAS for matmul operations
- [ ] Add stream/event management for pipelining
- [ ] Full forward/backward validation
- [ ] End-to-end benchmarks

### Phase 6: Optimization & Polish (Week 6)
- [ ] Profile and identify bottlenecks
- [ ] Tune block sizes and unroll factors
- [ ] Add PDL (Programmatic Dependent Launch) support where beneficial
- [ ] Documentation and examples
- [ ] Performance regression tests

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_cute_dsl/test_rmsnorm.py

import torch
import pytest
from mhc.cute_dsl import rmsnorm

class TestRMSNormDSL:
    @pytest.mark.parametrize("B,C", [(1, 64), (32, 256), (128, 1024)])
    def test_forward_matches_reference(self, B, C):
        inp = torch.randn(B, C, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(C, device='cuda', dtype=torch.bfloat16)
        eps = 1e-5
        
        # Reference implementation
        rms = torch.sqrt(inp.float().pow(2).mean(-1, keepdim=True) + eps)
        ref_out = (inp.float() / rms * weight.float()).bfloat16()
        
        # DSL implementation
        dsl_out = rmsnorm.rmsnorm_forward_dsl(inp, weight, eps)
        
        torch.testing.assert_close(dsl_out, ref_out, rtol=1e-2, atol=1e-3)
    
    def test_backward_gradients(self):
        B, C = 32, 256
        inp = torch.randn(B, C, device='cuda', dtype=torch.float32, 
                          requires_grad=True)
        weight = torch.randn(C, device='cuda', dtype=torch.float32,
                             requires_grad=True)
        
        # Use gradcheck
        torch.autograd.gradcheck(
            lambda x, w: rmsnorm.rmsnorm_dsl(x, w, eps=1e-5),
            (inp, weight),
            eps=1e-4
        )
```

### 7.2 Benchmark Suite

```python
# benchmarks/bench_cute_dsl.py

import torch
import time
from mhc.cute_dsl import rmsnorm, sinkhorn
from mhc import ops as cuda_ops

def benchmark_rmsnorm(B, C, warmup=10, iters=100):
    inp = torch.randn(B, C, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(C, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    for _ in range(warmup):
        _ = rmsnorm.rmsnorm_forward_dsl(inp, weight, 1e-5)
        _ = cuda_ops.rmsnorm(inp, weight, 1e-5)
    torch.cuda.synchronize()
    
    # Benchmark DSL
    start = time.perf_counter()
    for _ in range(iters):
        _ = rmsnorm.rmsnorm_forward_dsl(inp, weight, 1e-5)
    torch.cuda.synchronize()
    dsl_time = (time.perf_counter() - start) / iters * 1000
    
    # Benchmark CUDA
    start = time.perf_counter()
    for _ in range(iters):
        _ = cuda_ops.rmsnorm(inp, weight, 1e-5)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / iters * 1000
    
    print(f"B={B}, C={C}: DSL={dsl_time:.3f}ms, CUDA={cuda_time:.3f}ms, "
          f"ratio={dsl_time/cuda_time:.2f}x")
```

---

## 8. Known Challenges & Mitigations

### 8.1 Warp-Level Primitives

**Challenge**: CuTe DSL's warp shuffle support may differ from raw CUDA.

**Mitigation**: 
- Use `cute.shfl_xor`, `cute.shfl_down` if available
- Fall back to shared memory if needed
- Verify warp reduction correctness extensively

### 8.2 BFloat16 Vectorization

**Challenge**: bf16 vectorized loads (8 elements via float4) require careful type punning.

**Mitigation**:
- Use CuTe's native tensor layouts for vectorization
- Create helper functions for bf16 vector load/store
- May need to use explicit CUDA intrinsics via inline PTX if needed

### 8.3 cuBLAS Integration

**Challenge**: FusedRMSNormMatmul relies heavily on cuBLASLt.

**Mitigation**:
- Keep cuBLAS calls via PyTorch's linear layers or direct cuBLAS Python bindings
- Only port the custom kernel portions to CuTe DSL
- Use a hybrid approach for the layer

### 8.4 Dynamic Shared Memory

**Challenge**: Several kernels use `extern __shared__` for dynamic allocation.

**Mitigation**:
- Use CuTe DSL's `smem` kernel launch parameter
- Pre-calculate shared memory requirements in host code
- Document memory requirements clearly

### 8.5 Template Metaprogramming

**Challenge**: CUDA kernels use templates for compile-time constants (BLOCK_SIZE, MAX_N).

**Mitigation**:
- Use `cutlass.Constexpr` for compile-time values
- Use `cutlass.const_expr()` for compile-time conditionals
- Create specialized kernel variants as needed

---

## 9. Performance Targets

| Kernel | Target vs CUDA | Acceptable |
|--------|----------------|------------|
| RMSNorm Forward | 0.95x - 1.05x | 0.9x |
| RMSNorm Backward | 0.95x - 1.05x | 0.9x |
| Sinkhorn (64x64) | 0.90x - 1.00x | 0.85x |
| Stream Aggregate | 0.95x - 1.05x | 0.9x |
| Stream Mix Add | 0.95x - 1.05x | 0.9x |
| Full MHC Layer | 0.95x - 1.05x | 0.9x |

---

## 10. References

1. [CUTLASS Python DSL Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
2. [CuTe DSL Introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
3. [CuTe DSL Control Flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html)
4. [Framework Integration Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html)
5. Original MHC CUDA implementation in this repository

---

## Appendix A: Kernel Signature Reference

### A.1 Current CUDA Signatures

```cpp
// RMSNorm
void rmsnorm_forward(floatX* out, const floatX* inp, const floatX* weight, 
                     int N, int C, float eps, cudaStream_t stream);

void rmsnorm_forward_with_rms(floatX* out, float* rms_out, const floatX* inp,
                              const floatX* weight, int N, int C, float eps,
                              cudaStream_t stream);

void rmsnorm_backward(float* d_inp, float* d_weight, const float* grad, 
                      const floatX* inp, const floatX* weight, const float* rms,
                      int N, int C, cudaStream_t stream);

// Sinkhorn-Knopp
void sinkhorn_knopp_forward(float* out, const float* inp, int M, int N,
                            int num_iters, float eps, cudaStream_t stream);

void sinkhorn_knopp_backward(float* d_inp, const float* grad, const float* M_out,
                             const float* M_inp, int N, int num_iters, float eps,
                             cudaStream_t stream);

// Stream Operations
void stream_aggregate_bf16_fused_sigmoid(floatX* out, float* H_pre_activated,
                                         const float* inp, const float* H_pre_raw,
                                         int B, int n, int C, cudaStream_t stream);

void stream_distribute_mix_add_fused(float* out, float* H_post_activated,
                                     const float* x_inp, const floatX* y_norm,
                                     const float* H_post_raw, const float* M,
                                     int B, int n, int C, cudaStream_t stream);
```

### A.2 Proposed CuTe DSL Signatures

```python
# RMSNorm
@cute.jit
def rmsnorm_forward(out, inp, weight, N, C, eps, output_rms=False):
    """Launch RMSNorm forward kernel."""
    pass

@cute.jit
def rmsnorm_backward(d_inp, d_weight, grad, inp, weight, rms, N, C):
    """Launch RMSNorm backward kernel."""
    pass

# Sinkhorn-Knopp
@cute.jit
def sinkhorn_knopp_forward(out, inp, M, N, num_iters, eps):
    """Launch Sinkhorn-Knopp forward kernel."""
    pass

@cute.jit
def sinkhorn_knopp_backward(d_inp, grad, M_out, M_inp, N, num_iters, eps):
    """Launch Sinkhorn-Knopp backward kernel."""
    pass

# Stream Operations
@cute.jit
def stream_aggregate(out, H_pre_activated, inp, H_pre_raw, B, n, C):
    """Launch stream aggregation with fused sigmoid."""
    pass

@cute.jit
def stream_distribute_mix_add(out, H_post_activated, x_inp, y_norm,
                              H_post_raw, M, B, n, C):
    """Launch stream distribute with mix and add."""
    pass
```
