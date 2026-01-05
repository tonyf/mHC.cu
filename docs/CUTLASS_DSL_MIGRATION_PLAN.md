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

### 3.1 RMSNorm Forward (Quack-Style Implementation)

**Current CUDA Pattern**:
```cpp
template<int BLOCK_SIZE, bool OUTPUT_RMS>
__global__ void rmsnorm_kernel(floatX* out, float* rms_out, 
                               const floatX* inp, const floatX* weight,
                               int N, int C, float eps)
```

**CuTe DSL Implementation (Following Quack Patterns)**:

```python
from quack.reduce import row_reduce
from quack.copy_utils import tiled_copy_2d, predicate_k
from quack.reduction_base import ReductionBase

class RMSNorm(ReductionBase):
    """RMSNorm kernel using Quack patterns for speed-of-light performance."""
    
    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        super().__init__(dtype, N, stage=1)  # Single stage for RMSNorm
        self.reload_from = None if N <= 8192 else "smem"  # Reload strategy for large N

    def _threads_per_row(self):
        """Select optimal threads per row based on reduction dimension."""
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,           # [B, N] input
        mW: Optional[cute.Tensor], # [N] weight
        mO: cute.Tensor,           # [B, N] output
        mRstd: Optional[cute.Tensor],  # [B] rstd output
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self._set_cluster_n()
        largest_dtype_width = max(t.element_type.width for t in [mX, mW, mO] if t is not None)
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        
        self.kernel(mX, mW, mO, mRstd, eps, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[tiled_copy.size, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mO: cute.Tensor,
        mRstd: Optional[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = cute.arch.block_idx()[1] if self.cluster_n > 1 else 0
        tv_layout = tiled_copy.layout_tv_tiled

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, 
            cute.make_ordered_layout(tiler_mn, order=(1, 0)), 
            byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        
        # Partition for this CTA
        gX, gO, gRstd, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mO, mRstd, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None

        thr_copy_X = tiled_copy.get_slice(tidx)
        
        # Partition tensors for this thread
        tXgW = thr_copy_X.partition_S(gW) if mW is not None else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        
        # Allocate register fragments
        tXrW = cute.make_fragment_like(tXgW) if mW is not None else None
        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]

        # Initialize cluster if needed
        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        # Handle uneven dimensions
        is_even_N = shape[1] == tiler_mn[1] * self.cluster_n
        tXpX = predicate_k(thr_copy_X.partition_S(cX), limit=shape[1]) if not is_even_N else None
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        # Load weights while waiting for data
        if mW is not None:
            copy(tXgW, tXrW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        # Compute sum of squares with row reduction (supports cluster for large N)
        sum_sq_x = row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if self.cluster_n > 1 else None,
        )
        
        # Compute rstd
        rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)
        
        # Store rstd if requested
        if mRstd is not None:
            if tXcX[0][1] == 0 and row < shape[0]:
                if self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0:
                    thr_copy_X.partition_D(gRstd)[0] = rstd

        # Reload x if needed for large N
        if self.reload_from == "smem":
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)

        # Normalize and apply weight
        o = x * rstd
        if mW is not None:
            w = tXrW.load().to(cute.Float32)
            o = o * w
        
        tXrO.store(o.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)
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

## 4. Reusable Components (Quack-Style)

### 4.1 Core Utilities Module (`utils.py`)

Adapted from `quack/utils.py`:

```python
"""Core utilities for CuTe DSL kernels - adapted from Quack."""

from functools import partial
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm, vector

# Packed f32 operations with proper rounding
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)

@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

@cute.jit
def load_scalar_or_pointer(x: Float32 | cute.Pointer) -> Float32:
    """Load scalar value or dereference pointer."""
    if const_expr(isinstance(x, cute.Pointer)):
        return Float32(cute.make_tensor(x, cute.make_layout(1))[0])
    else:
        return x

@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, 
    peer_cta_rank_in_cluster: Int32, 
    *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(llvm.inline_asm(
        T.i32(),
        [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))

@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *, loc=None, ip=None,
) -> None:
    """Store to another CTA's shared memory via distributed shared memory."""
    remote_smem_ptr_i32 = set_block_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    suffix = {Float32: "f32", Int32: "s32"}[type(val)]
    constraint = {Float32: "f", Int32: "r"}[type(val)]
    
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    """Pack two f32 values into i64 for efficient smem transfers."""
    vec_f32x2 = vector.from_elements(T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip)
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    return cutlass.Int64(vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip))

@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Unpack i64 back to two f32 values."""
    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip))
    res1 = Float32(vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip))
    return res0, res1

@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor."""
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])

@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Atomic add for int32."""
    return nvvm.atomicrmw(res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value())
```

### 4.2 Reduction Module (`reduce.py`)

Adapted from `quack/reduce.py`:

```python
"""Reduction operations - adapted from Quack."""

import math
import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

from .utils import elem_pointer, store_shared_remote, f32x2_to_i64, i64_to_f32x2

@cute.jit
def block_reduce(
    val: cute.Numeric, 
    op: Callable, 
    reduction_buffer: cute.Tensor, 
    init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    """Block reduction via shared memory.
    
    Args:
        val: Per-thread value to reduce
        op: Reduction operator (add, max, etc.)
        reduction_buffer: Shared memory buffer shape (num_warps/warps_per_row, warps_per_row)
        init_val: Initial value for reduction
    """
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)

@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
) -> cute.Numeric:
    """Cluster reduction via distributed shared memory.
    
    Args:
        val: Per-thread value to reduce
        op: Reduction operator
        reduction_buffer: Shape (num_warps/warps_per_row, (warps_per_row, cluster_n))
        mbar_ptr: Memory barrier pointer
        init_val: Initial value
        phase: Barrier phase
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    
    return cute.arch.warp_reduction(block_reduce_val, op)

@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """Unified row reduction supporting warp, block, and cluster levels.
    
    Args:
        x: Input tensor or scalar
        op: Reduction operation (ADD, MAX, MIN, MUL)
        threads_per_row: Number of threads participating per row
        reduction_buffer: Shared memory for block/cluster reduction
        mbar_ptr: Memory barrier for cluster reduction
        phase: Barrier phase
        init_val: Initial value
        hook_fn: Hook function called between warp and block reduction
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    
    val = cute.arch.warp_reduction(
        val, warp_op, 
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE)
    )
    
    if const_expr(hook_fn is not None):
        hook_fn()
    
    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            if const_expr(mbar_ptr is None):
                val = block_reduce(val, warp_op, reduction_buffer, init_val=init_val)
            else:
                val = cluster_reduce(val, warp_op, reduction_buffer, mbar_ptr, 
                                    phase=phase, init_val=init_val)
    return val
```

### 4.3 Copy Utilities Module (`copy_utils.py`)

Adapted from `quack/copy_utils.py`:

```python
"""Copy and memory utilities - adapted from Quack."""

from typing import Optional, Type
from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync

@cute.jit
def load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    """Load from shared memory to registers."""
    dst = cute.make_fragment_like(src, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst

def tiled_copy_1d(
    dtype: Type[cutlass.Numeric], 
    num_threads: int, 
    num_copy_elems: int = 1, 
    is_async: bool = False
) -> cute.TiledCopy:
    """Create 1D tiled copy pattern."""
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    """Create 2D tiled copy for row-major data with vectorized loads."""
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    """Compute predicates for K-dimension bounds checking."""
    tApA = cute.make_fragment(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA

def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    **kwargs,
) -> None:
    """Generic copy with optional predication and async support."""
    num_copy_elems = src.shape[0][0]
    num_copy_bits = min(128, num_copy_elems * src.element_type.width)
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, src.element_type, num_bits_per_copy=num_copy_bits)
    cute.copy(copy_atom, src, dst, pred=pred, **kwargs)
```

### 4.4 Fast Math Module (`fast_math.py`)

Adapted from `quack/fast_math.py`:

```python
"""Fast math utilities - adapted from Quack."""

from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .cute_dsl_utils import ParamsBase

@cute.jit
def clz(x: Int32) -> Int32:
    """Count leading zeros."""
    res = Int32(32)
    done = False
    for i in cutlass.range(32):
        if ((1 << (31 - i)) & x) and not done:
            res = Int32(i)
            done = True
    return res

def find_log2(x: Int32) -> Int32:
    """Find log2 rounded up."""
    a = Int32(31 - clz(x))
    return a + ((x & (x - 1)) != 0)

@dsl_user_op
def umulhi(a: Int32, b: Int32, *, loc=None, ip=None) -> Uint32:
    """Unsigned multiply high - returns high 32 bits of 64-bit product."""
    return Uint32(llvm.inline_asm(
        T.i32(),
        [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
        "mul.hi.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))

@dataclass
class FastDivmod(ParamsBase):
    """Fast integer division using multiply-shift technique.
    
    Precompute division parameters on host, then use fast multiply-shift on device.
    """
    divisor: Int32
    multiplier: Uint32
    shift_right: Uint32

    @staticmethod
    def create(divisor: Int32) -> "FastDivmod":
        """Precompute fast divmod parameters (call on host)."""
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        """Fast division on device."""
        return Int32(umulhi(dividend, self.multiplier) >> self.shift_right) \
               if self.divisor != 1 else dividend

    def divmod(self, dividend: Int32) -> Tuple[Int32, Int32]:
        """Fast divmod on device."""
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder
```

### 4.5 Reduction Base Class (`base.py`)

Adapted from `quack/reduction_base.py`:

```python
"""Base class for reduction kernels - adapted from Quack."""

from typing import Type, Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, const_expr

from . import copy_utils

class ReductionBase:
    """Base class for reduction kernels (RMSNorm, Softmax, etc.)."""
    
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, stage: int = 1, 
                 reduction_dtype=Float32):
        self.dtype = dtype
        self.N = N
        self.stage = stage  # For double-buffering (LayerNorm needs 2)
        self.reduction_dtype = reduction_dtype
        self.cluster_n = 1

    def _threads_per_row(self) -> int:
        """Select optimal threads per row based on reduction dimension N."""
        raise NotImplementedError()

    def _num_threads(self) -> int:
        """Select total thread count."""
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        """Set cluster size based on N (for distributed shared memory reduction)."""
        self.cluster_n = 1  # Override in subclass for large N

    def _get_tiled_copy(self, vecsize: int = 1):
        """Create tiled copy configuration for this kernel."""
        assert self.N % vecsize == 0
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0
        
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout: cute.Layout, cluster_n: int):
        """Compute reduction buffer layout based on thread-value layout."""
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps if cute.rank(tv_layout.shape[0]) == 1 
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _allocate_reduction_buffer_and_mbar(
        self, 
        smem: cutlass.utils.SmemAllocator, 
        tv_layout: cute.Layout
    ) -> Tuple[cute.Tensor, Optional[cute.Pointer]]:
        """Allocate reduction buffer and optional cluster barrier."""
        reduction_buffer = smem.allocate_tensor(
            self.reduction_dtype,
            self._get_reduction_buffer_layout(tv_layout, self.cluster_n),
            byte_alignment=8,
        )
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage)
        else:
            mbar_ptr = None
        return reduction_buffer, mbar_ptr

    @cute.jit
    def _initialize_cluster(self, tidx: Int32, mbar_ptr: cute.Pointer, num_warps: int):
        """Initialize cluster barriers if using distributed shared memory."""
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
```

---

## 5. PyTorch Integration Strategy (Quack-Style)

### 5.1 Kernel Compilation with Caching

Following Quack's pattern, use `cute.compile` with fake tensors and cache compiled kernels:

```python
"""PyTorch integration for CuTe DSL kernels - following Quack pattern."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
import cutlass
import cutlass.cute as cute
from cutlass import Float32

from .cute_dsl.rmsnorm import RMSNorm
from .cute_dsl.compile_utils import make_fake_tensor as fake_tensor
from .cute_dsl.cute_dsl_utils import torch2cute_dtype_map

# Dtype mapping
torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}

def _rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    rstd: Optional[Tensor],
    eps: float,
):
    """Internal forward implementation with compilation caching."""
    B, N = x.shape
    dtype = torch2cute_dtype_map[x.dtype]
    weight_dtype = torch2cute_dtype_map[weight.dtype] if weight is not None else None
    out_dtype = torch2cute_dtype_map[out.dtype]
    
    # Create compile key from shapes and dtypes (shapes don't matter, only dtypes)
    compile_key = (dtype, weight_dtype, out_dtype, N)
    
    if compile_key not in _rmsnorm_fwd.compile_cache:
        # Create fake tensors for compilation (shape-agnostic via batch_sym)
        batch_sym = cute.symbolic.make_int_symbol()
        all_dtypes = [dtype, out_dtype, weight_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        
        x_cute = fake_tensor(dtype, (batch_sym, N), div)
        out_cute = fake_tensor(out_dtype, (batch_sym, N), div)
        weight_cute = fake_tensor(weight_dtype, (N,), div) if weight_dtype else None
        rstd_cute = fake_tensor(Float32, (batch_sym,)) if rstd is not None else None
        
        # Compile kernel with TVM FFI for PyTorch stream integration
        _rmsnorm_fwd.compile_cache[compile_key] = cute.compile(
            RMSNorm(dtype, N),
            x_cute,
            weight_cute,
            out_cute,
            rstd_cute,
            Float32(0),  # eps placeholder
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    
    # Call compiled kernel
    _rmsnorm_fwd.compile_cache[compile_key](x, weight, out, rstd, eps)

_rmsnorm_fwd.compile_cache = {}


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """RMSNorm forward with automatic output allocation."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32) if store_rstd else None
    
    _rmsnorm_fwd(x, weight, out, rstd, eps)
    return out, rstd
```

### 5.2 Autograd Function with Compilation Caching

```python
from torch.autograd import Function

class RMSNormFunction(Function):
    @staticmethod
    def forward(ctx, inp: Tensor, weight: Optional[Tensor], eps: float):
        out, rstd = rmsnorm_fwd(inp, weight, eps=eps, store_rstd=True)
        ctx.save_for_backward(inp, weight, rstd)
        ctx.eps = eps
        return out
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inp, weight, rstd = ctx.saved_tensors
        
        d_inp = torch.empty_like(inp)
        d_weight = torch.zeros_like(weight) if weight is not None else None
        
        _rmsnorm_bwd(grad_output, inp, weight, rstd, d_inp, d_weight)
        
        return d_inp, d_weight, None

def rmsnorm(inp: Tensor, weight: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """Functional interface for RMSNorm."""
    return RMSNormFunction.apply(inp, weight, eps)
```

### 5.3 Fake Tensor Utilities for Compilation

```python
"""Compilation utilities - adapted from Quack."""

import cutlass
import cutlass.cute as cute

def make_fake_tensor(dtype, shape, alignment_divisor=1):
    """Create fake tensor for kernel compilation.
    
    Args:
        dtype: CuTe dtype (Float32, BFloat16, etc.)
        shape: Tensor shape (can include symbolic dimensions)
        alignment_divisor: For vectorized loads, ensure shape is divisible by this
    
    Returns:
        Fake CuTe tensor suitable for compilation
    """
    if dtype is None:
        return None
    
    # Create layout
    layout = cute.make_layout(shape)
    
    # Create fake pointer with proper alignment
    fake_ptr = cute.make_ptr(dtype, 0, cute.AddressSpace.gmem)
    
    # Create tensor
    tensor = cute.make_tensor(fake_ptr, layout)
    
    # Mark dimensions as dynamic for shape-agnostic compilation
    tensor = tensor.mark_layout_dynamic()
    
    return tensor
```

### 5.4 Stream Integration

```python
"""Stream management for kernel launches."""

import torch
import cuda.bindings.driver as cuda

def get_cuda_stream() -> cuda.CUstream:
    """Get current CUDA stream from PyTorch."""
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

# In kernel class __call__ method:
# stream = get_cuda_stream() if stream is None else stream
# self.kernel(...).launch(..., stream=stream)
```

---

## 6. Migration Task List

### Phase 0: Quack Integration Decision
**Option A: Direct Dependency**
- Add `quack-kernels` as a project dependency
- Directly use Quack's RMSNorm forward/backward implementations
- Focus MHC-specific kernels on Sinkhorn-Knopp and Stream operations

**Option B: Adapt Patterns (Recommended)**
- Copy and adapt key utility modules from Quack
- Maintain full control over implementation
- Customize for MHC-specific requirements

### Phase 1: Infrastructure (Week 1)
- [ ] Set up `cute_dsl/` module structure
- [ ] Copy and adapt `utils.py` from Quack (elem_pointer, store_shared_remote, f32x2 packing)
- [ ] Copy and adapt `reduce.py` from Quack (block_reduce, cluster_reduce, row_reduce)
- [ ] Copy and adapt `copy_utils.py` from Quack (tiled_copy_2d, predicate_k)
- [ ] Copy and adapt `fast_math.py` from Quack (FastDivmod)
- [ ] Implement `base.py` ReductionBase class
- [ ] Create `cute_dsl_utils.py` with ParamsBase, ArgumentsBase, dtype mapping
- [ ] Create test infrastructure with PyTorch reference implementations
- [ ] Add `quack-kernels` as optional dependency for comparison benchmarks

### Phase 2: RMSNorm (Week 2)
- [ ] Implement `RMSNorm` class following Quack pattern
  - [ ] Forward kernel with SmemAllocator
  - [ ] Tiled copy with vectorized loads
  - [ ] row_reduce with cluster support for large N (>16k)
  - [ ] Reload strategy for very large N (>8k)
- [ ] Implement `RMSNormBackward` class
  - [ ] Persistent kernel pattern for backward
  - [ ] Double-buffering for weight gradient accumulation
- [ ] Add PyTorch autograd wrapper using `cute.compile` caching
- [ ] Validate against existing CUDA implementation AND Quack
- [ ] Benchmark vs Quack and original CUDA (target: match Quack performance)

### Phase 3: Sinkhorn-Knopp (Week 3)
- [ ] Design Sinkhorn kernel class structure (no Quack equivalent)
- [ ] Implement `sinkhorn_knopp_single_block_kernel` (M,N ≤ 64)
  - [ ] Use SmemAllocator for tile storage
  - [ ] Iterative row/column normalization with warp reductions
  - [ ] Fast reciprocal via `cute.math.rsqrt` or PTX `frcp`
- [ ] Implement `sinkhorn_knopp_warp_optimized_kernel` (32x32 special case)
  - [ ] Leverage warp-level primitives for 32-wide rows/cols
- [ ] Implement `sinkhorn_knopp_batched_kernel`
  - [ ] Multi-block parallel batches
- [ ] Implement `sinkhorn_knopp_backward_kernel`
  - [ ] Forward checkpoint recomputation
- [ ] Add fused exp variant
- [ ] Validate and benchmark

### Phase 4: Stream Operations (Week 4)
- [ ] Design Stream kernel class structure
- [ ] Implement `StreamAggregate` class
  - [ ] Fused sigmoid activation
  - [ ] Vectorized bf16 accumulation
  - [ ] Dynamic n variants
- [ ] Implement `StreamDistributeMixAdd` class
  - [ ] Fused activations on H_post
  - [ ] Shared memory for weight broadcast
- [ ] Implement backward kernels
  - [ ] Multi-output gradient computation
- [ ] Validate and benchmark

### Phase 5: Integration (Week 5)
- [ ] Create unified `MHCLayerDSL` class
- [ ] Integrate with cuBLAS for matmul operations
  - [ ] Wrap in Python using PyTorch GEMM or direct cuBLAS bindings
- [ ] Add stream/event management for pipelining
- [ ] Implement `cute.compile` caching for all kernels
- [ ] Full forward/backward validation
- [ ] End-to-end benchmarks

### Phase 6: Optimization & Polish (Week 6)
- [ ] Profile with Nsight Compute
- [ ] Compare against theoretical memory bandwidth limits
- [ ] Tune:
  - [ ] threads_per_row thresholds
  - [ ] cluster_n thresholds
  - [ ] vecsize selection
  - [ ] reload_from strategy
- [ ] Add PDL (Programmatic Dependent Launch) support where beneficial
- [ ] Documentation and examples
- [ ] Performance regression tests
- [ ] Consider contributing improvements back to Quack

---

## 7. Testing Strategy with pytest

### 7.1 Design Principles for Testing

#### Correctness Testing Principles

1. **Reference Implementation Comparison**: Every kernel must have a PyTorch reference implementation for ground truth
2. **Tolerance-Based Assertions**: Use dtype-appropriate tolerances (bf16: `atol=1e-1`, fp16: `atol=1e-2`, fp32: `atol=1e-4`)
3. **Parametric Coverage**: Test across multiple dimensions, batch sizes, and dtypes using `@pytest.mark.parametrize`
4. **Boundary Conditions**: Test edge cases (N=1, non-power-of-2 dimensions, uneven shapes)
5. **Numerical Stability**: Test with extreme values (large, small, mixed) to verify stability
6. **Gradient Verification**: Use `torch.autograd.gradcheck` for backward pass validation
7. **Property Testing**: Verify mathematical properties (e.g., doubly stochastic for Sinkhorn)

#### Performance Testing Principles

1. **Warmup Runs**: Always execute warmup iterations before timing
2. **Multiple Iterations**: Average over many runs to reduce noise
3. **CUDA Synchronization**: Call `torch.cuda.synchronize()` before/after timing
4. **Baseline Comparison**: Compare against both original CUDA and Quack implementations
5. **Memory Bandwidth Analysis**: Calculate achieved bandwidth vs theoretical peak
6. **Regression Detection**: Track performance over time, fail CI on significant regression

### 7.2 Test File Organization

```
tests/
├── conftest.py                      # Shared fixtures and utilities
├── test_cute_dsl/
│   ├── __init__.py
│   ├── test_utils.py                # Test utility modules
│   ├── test_reduce.py               # Test reduction primitives
│   ├── test_copy_utils.py           # Test copy utilities
│   ├── test_rmsnorm.py              # RMSNorm forward/backward tests
│   ├── test_sinkhorn.py             # Sinkhorn-Knopp tests
│   ├── test_stream_ops.py           # Stream operations tests
│   ├── test_mhc_layer.py            # Full layer integration tests
│   └── test_compile_cache.py        # JIT compilation caching tests
├── benchmarks/
│   ├── __init__.py
│   ├── bench_rmsnorm.py
│   ├── bench_sinkhorn.py
│   ├── bench_stream_ops.py
│   └── bench_mhc_layer.py
```

### 7.3 Shared Test Fixtures (`conftest.py`)

```python
"""Shared pytest fixtures for CuTe DSL tests."""

import pytest
import torch

# Increase torch.compile cache for parametric tests
torch._dynamo.config.cache_size_limit = 1024
torch._dynamo.config.accumulated_cache_size_limit = 1024


@pytest.fixture(scope="session")
def device():
    """CUDA device fixture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    return 42


def get_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Get appropriate tolerance for dtype."""
    if dtype == torch.bfloat16:
        return 1e-1, 1e-2  # atol, rtol
    elif dtype == torch.float16:
        return 1e-2, 1e-3
    elif dtype == torch.float32:
        return 1e-4, 1e-4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def rmsnorm_ref(x, weight=None, bias=None, residual=None, eps=1e-6):
    """Reference implementation for RMSNorm."""
    x_f32 = x.float()
    if residual is not None:
        x_f32 = x_f32 + residual.float()
    rms = torch.sqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    x_norm = x_f32 / rms
    out = x_norm * weight.float() if weight is not None else x_norm
    if bias is not None:
        out = out + bias.float()
    if residual is None:
        return out.to(x.dtype)
    else:
        return out.to(x.dtype), x_f32.to(residual.dtype)


def sinkhorn_ref(M, num_iters=10, eps=1e-8):
    """Reference implementation for Sinkhorn-Knopp."""
    M = M.float()
    for _ in range(num_iters):
        M = M / (M.sum(dim=1, keepdim=True) + eps)
        M = M / (M.sum(dim=0, keepdim=True) + eps)
    return M
```

### 7.4 RMSNorm Test Suite (`test_rmsnorm.py`)

```python
"""RMSNorm kernel tests - following Quack testing patterns."""

import pytest
import torch

from mhc.cute_dsl.rmsnorm import rmsnorm, rmsnorm_fwd, _rmsnorm_fwd
from tests.conftest import get_tolerance, rmsnorm_ref


class TestRMSNormForward:
    """Forward pass correctness tests."""
    
    @pytest.mark.parametrize("eps", [1e-5, 1e-6])
    @pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float16, torch.float32, None])
    @pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16, torch.float32])
    @pytest.mark.parametrize("N", [
        64, 128, 256, 512, 1024, 2048, 4096,  # Small to medium
        8192, 16384,                           # Large (single block)
        32768, 65536, 131072,                  # Very large (cluster reduction)
    ])
    @pytest.mark.parametrize("M", [1, 37, 199, 1024])  # Include non-power-of-2
    @pytest.mark.parametrize("use_compile", [False, True])
    def test_forward_correctness(self, M, N, input_dtype, weight_dtype, eps, use_compile, device, seed):
        """Test forward pass matches reference implementation."""
        # Skip OOM-prone combinations
        if N >= 128 * 1024 and input_dtype == torch.float32 and M >= 1024:
            pytest.skip("Skipping large float32 test to avoid OOM")
        
        atol, rtol = get_tolerance(input_dtype)
        
        x = torch.randn(M, N, device=device, dtype=input_dtype, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=weight_dtype) if weight_dtype else None
        
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone() if weight is not None else None
        
        fn = torch.compile(rmsnorm, fullgraph=True) if use_compile else rmsnorm
        out = fn(x, weight, eps=eps)
        out_ref = rmsnorm_ref(x_ref, weight_ref, eps=eps)
        
        assert out.shape == x.shape
        assert out.dtype == input_dtype
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("N", [192, 760, 1128, 3000])  # Non-power-of-2 dimensions
    def test_uneven_dimensions(self, N, device, seed):
        """Test with non-power-of-2 dimensions."""
        M = 32
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        weight = torch.randn(N, device=device, dtype=torch.float32)
        
        out = rmsnorm(x, weight)
        out_ref = rmsnorm_ref(x, weight)
        
        torch.testing.assert_close(out, out_ref, atol=1e-1, rtol=1e-2)


class TestRMSNormBackward:
    """Backward pass correctness tests."""
    
    @pytest.mark.parametrize("N", [256, 1024, 4096, 16384])
    @pytest.mark.parametrize("M", [1, 32, 128])
    def test_backward_correctness(self, M, N, device, seed):
        """Test backward pass matches reference implementation."""
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)
        
        x_ref = x.detach().clone().requires_grad_()
        weight_ref = weight.detach().clone().requires_grad_()
        
        out = rmsnorm(x, weight)
        out_ref = rmsnorm_ref(x_ref, weight_ref)
        
        grad_out = torch.randn_like(out)
        torch.cuda.synchronize()
        
        out.backward(grad_out)
        out_ref.backward(grad_out)
        
        atol, rtol = get_tolerance(x.dtype)
        torch.testing.assert_close(x.grad, x_ref.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(weight.grad, weight_ref.grad, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.float32])  # gradcheck requires float32
    def test_gradcheck(self, dtype, device, seed):
        """Test gradients with torch.autograd.gradcheck."""
        M, N = 8, 64  # Small for gradcheck
        x = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
        
        def fn(x, w):
            return rmsnorm(x, w, eps=1e-5)
        
        torch.autograd.gradcheck(fn, (x, weight), eps=1e-4, atol=1e-3, rtol=1e-3)


class TestRMSNormNumericalStability:
    """Numerical stability tests."""
    
    def test_large_values(self, device, seed):
        """Test with large input values."""
        M, N = 32, 1024
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16) * 1000
        weight = torch.randn(N, device=device, dtype=torch.float32)
        
        out = rmsnorm(x, weight)
        
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_small_values(self, device, seed):
        """Test with small input values."""
        M, N = 32, 1024
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16) * 1e-6
        weight = torch.randn(N, device=device, dtype=torch.float32)
        
        out = rmsnorm(x, weight)
        
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_mixed_extreme_values(self, device, seed):
        """Test with mixed large and small values."""
        M, N = 32, 1024
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        x[:, :N//2] *= 1000
        x[:, N//2:] *= 1e-6
        weight = torch.randn(N, device=device, dtype=torch.float32)
        
        out = rmsnorm(x, weight)
        
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"


class TestRMSNormCompileCache:
    """JIT compilation cache tests."""
    
    def test_cache_reuse_same_dtype(self, device, seed):
        """Test cache reuse for same dtype, different batch size."""
        _rmsnorm_fwd.compile_cache.clear()
        
        N = 1024
        weight = torch.randn(N, device=device, dtype=torch.float32)
        
        # First call
        x1 = torch.randn(32, N, device=device, dtype=torch.bfloat16)
        rmsnorm_fwd(x1, weight)
        cache_size_1 = len(_rmsnorm_fwd.compile_cache)
        
        # Different batch size - should reuse cache
        x2 = torch.randn(64, N, device=device, dtype=torch.bfloat16)
        rmsnorm_fwd(x2, weight)
        cache_size_2 = len(_rmsnorm_fwd.compile_cache)
        
        assert cache_size_1 == cache_size_2, "Cache should be reused for different batch sizes"

    def test_cache_miss_different_n(self, device, seed):
        """Test cache miss for different N dimension."""
        _rmsnorm_fwd.compile_cache.clear()
        
        M = 32
        
        # First N
        x1 = torch.randn(M, 1024, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(1024, device=device, dtype=torch.float32)
        rmsnorm_fwd(x1, w1)
        cache_size_1 = len(_rmsnorm_fwd.compile_cache)
        
        # Different N - should create new cache entry
        x2 = torch.randn(M, 2048, device=device, dtype=torch.bfloat16)
        w2 = torch.randn(2048, device=device, dtype=torch.float32)
        rmsnorm_fwd(x2, w2)
        cache_size_2 = len(_rmsnorm_fwd.compile_cache)
        
        assert cache_size_2 == cache_size_1 + 1, "Different N should create new cache entry"


class TestRMSNormInputValidation:
    """Input validation tests."""
    
    def test_weight_dimension_mismatch(self, device):
        """Test error on weight dimension mismatch."""
        x = torch.randn(32, 1024, device=device, dtype=torch.bfloat16)
        weight = torch.randn(512, device=device, dtype=torch.float32)  # Wrong size
        
        with pytest.raises((ValueError, RuntimeError)):
            rmsnorm(x, weight)

    def test_cpu_tensor_rejected(self, device):
        """Test that CPU tensors are rejected."""
        x = torch.randn(32, 1024, dtype=torch.bfloat16)  # CPU
        weight = torch.randn(1024, dtype=torch.float32)
        
        with pytest.raises((AssertionError, NotImplementedError, RuntimeError)):
            rmsnorm(x, weight)

    def test_unsupported_dtype(self, device):
        """Test that unsupported dtypes are rejected."""
        x = torch.randn(32, 1024, device=device, dtype=torch.float64)
        weight = torch.randn(1024, device=device, dtype=torch.float32)
        
        with pytest.raises((AssertionError, ValueError, KeyError)):
            rmsnorm(x, weight)
```

### 7.5 Sinkhorn-Knopp Test Suite (`test_sinkhorn.py`)

```python
"""Sinkhorn-Knopp kernel tests."""

import pytest
import torch

from mhc.cute_dsl.sinkhorn import sinkhorn_knopp
from tests.conftest import sinkhorn_ref


class TestSinkhornForward:
    """Forward pass correctness tests."""
    
    @pytest.mark.parametrize("size", [8, 16, 32, 48, 64])  # Single block sizes
    @pytest.mark.parametrize("num_iters", [5, 10, 20, 50])
    def test_doubly_stochastic_property(self, size, num_iters, device, seed):
        """Test output is doubly stochastic matrix."""
        M = torch.rand(size, size, device=device) + 0.1
        
        out = sinkhorn_knopp(M, num_iters=num_iters)
        
        row_sums = out.sum(dim=1)
        col_sums = out.sum(dim=0)
        
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)
        assert (out >= 0).all(), "Output should be non-negative"

    @pytest.mark.parametrize("size", [16, 32, 64])
    def test_matches_reference(self, size, device, seed):
        """Test output matches reference implementation."""
        M = torch.rand(size, size, device=device) + 0.1
        
        out = sinkhorn_knopp(M, num_iters=20)
        out_ref = sinkhorn_ref(M, num_iters=20)
        
        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("M,N", [(16, 32), (32, 16), (24, 48)])  # Non-square
    def test_non_square_matrices(self, M, N, device, seed):
        """Test with non-square matrices."""
        mat = torch.rand(M, N, device=device) + 0.1
        
        out = sinkhorn_knopp(mat, num_iters=20)
        
        assert out.shape == (M, N)
        assert not torch.isnan(out).any()


class TestSinkhornBackward:
    """Backward pass tests."""
    
    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_gradient_exists(self, size, device, seed):
        """Test that gradients flow through."""
        M = (torch.rand(size, size, device=device) + 0.1).requires_grad_(True)
        
        out = sinkhorn_knopp(M, num_iters=10)
        loss = out.sum()
        loss.backward()
        
        assert M.grad is not None
        assert not torch.isnan(M.grad).any()
        assert not torch.isinf(M.grad).any()

    def test_gradcheck(self, device, seed):
        """Test gradients with torch.autograd.gradcheck."""
        size = 8
        M = (torch.rand(size, size, device=device, dtype=torch.float64) + 0.1).requires_grad_(True)
        
        def fn(x):
            return sinkhorn_knopp(x, num_iters=5)
        
        torch.autograd.gradcheck(fn, (M,), eps=1e-4)


class TestSinkhornNumericalStability:
    """Numerical stability tests."""
    
    def test_large_values(self, device, seed):
        """Test stability with large input values."""
        size = 32
        M = torch.rand(size, size, device=device) * 1000
        
        out = sinkhorn_knopp(M, num_iters=50)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_small_values(self, device, seed):
        """Test stability with small input values."""
        size = 32
        M = torch.rand(size, size, device=device) * 1e-6 + 1e-8
        
        out = sinkhorn_knopp(M, num_iters=20)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_sparse_input(self, device, seed):
        """Test with sparse-like input (many near-zero values)."""
        size = 32
        M = torch.rand(size, size, device=device)
        M[M < 0.9] = 1e-8  # Make most values very small
        
        out = sinkhorn_knopp(M, num_iters=20)
        
        assert not torch.isnan(out).any()


class TestSinkhornBatched:
    """Batched operation tests."""
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batched_forward(self, batch_size, device, seed):
        """Test batched forward pass."""
        size = 32
        M = torch.rand(batch_size, size, size, device=device) + 0.1
        
        out = sinkhorn_knopp(M, num_iters=20)
        
        assert out.shape == (batch_size, size, size)
        
        # Check each batch element is doubly stochastic
        for i in range(batch_size):
            row_sums = out[i].sum(dim=1)
            col_sums = out[i].sum(dim=0)
            torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4)
```

### 7.6 Stream Operations Test Suite (`test_stream_ops.py`)

```python
"""Stream operations kernel tests."""

import pytest
import torch

from mhc.cute_dsl.stream_ops import stream_aggregate, stream_distribute_mix_add


class TestStreamAggregate:
    """Stream aggregation tests."""
    
    @pytest.mark.parametrize("B", [1, 8, 32])
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    @pytest.mark.parametrize("C", [64, 128, 256, 1024])
    def test_forward_shape(self, B, n, C, device, seed):
        """Test output shape is correct."""
        inp = torch.randn(B, n, C, device=device, dtype=torch.bfloat16)
        H_pre = torch.randn(n, device=device, dtype=torch.float32)
        
        out, H_activated = stream_aggregate(inp, H_pre)
        
        assert out.shape == (B, C)
        assert H_activated.shape == (n,)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_sigmoid_activation(self, n, device, seed):
        """Test H weights are properly sigmoid-activated."""
        B, C = 8, 128
        inp = torch.randn(B, n, C, device=device, dtype=torch.bfloat16)
        H_pre = torch.randn(n, device=device, dtype=torch.float32)
        
        _, H_activated = stream_aggregate(inp, H_pre)
        
        # Activated weights should be in (0, 1) due to sigmoid
        assert (H_activated > 0).all()
        assert (H_activated < 1).all()
        
        # Check against reference sigmoid
        H_ref = torch.sigmoid(H_pre)
        torch.testing.assert_close(H_activated, H_ref, atol=1e-5, rtol=1e-5)


class TestStreamDistributeMixAdd:
    """Stream distribute/mix/add tests."""
    
    @pytest.mark.parametrize("B", [1, 8, 32])
    @pytest.mark.parametrize("n", [2, 4, 8])
    @pytest.mark.parametrize("C", [64, 128, 256])
    def test_forward_shape(self, B, n, C, device, seed):
        """Test output shape is correct."""
        x = torch.randn(B, C, device=device, dtype=torch.float32)
        y_norm = torch.randn(B, C, device=device, dtype=torch.bfloat16)
        H_post = torch.randn(n, device=device, dtype=torch.float32)
        M = torch.randn(n, n, device=device, dtype=torch.float32)
        
        out, H_activated = stream_distribute_mix_add(x, y_norm, H_post, M)
        
        assert out.shape == (B, n, C)
        assert H_activated.shape == (n,)


class TestStreamOpsBackward:
    """Backward pass tests for stream operations."""
    
    def test_aggregate_backward(self, device, seed):
        """Test backward pass for stream_aggregate."""
        B, n, C = 8, 4, 128
        inp = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)
        H_pre = torch.randn(n, device=device, dtype=torch.float32, requires_grad=True)
        
        out, _ = stream_aggregate(inp, H_pre)
        loss = out.sum()
        loss.backward()
        
        assert inp.grad is not None
        assert H_pre.grad is not None
        assert not torch.isnan(inp.grad).any()
        assert not torch.isnan(H_pre.grad).any()
```

### 7.7 Integration Test Suite (`test_mhc_layer.py`)

```python
"""Full MHC layer integration tests."""

import pytest
import torch

from mhc.cute_dsl.mhc_layer import MHCLayerDSL


class TestMHCLayerForward:
    """Forward pass integration tests."""
    
    @pytest.mark.parametrize("B", [1, 8, 32])
    @pytest.mark.parametrize("n", [2, 4, 8])
    @pytest.mark.parametrize("C", [64, 128, 256])
    def test_forward_shape(self, B, n, C, device, seed):
        """Test full layer forward produces correct shape."""
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device)
        
        out = layer(x)
        
        assert out.shape == (B, n, C)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic(self, device, seed):
        """Test layer is deterministic."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device)
        
        out1 = layer(x)
        out2 = layer(x)
        
        torch.testing.assert_close(out1, out2)


class TestMHCLayerBackward:
    """Backward pass integration tests."""
    
    def test_gradient_flow(self, device, seed):
        """Test gradients flow through all parameters."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device, requires_grad=True)
        
        out = layer(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.norm() > 0, "Input gradient should be non-zero"
        
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"


class TestMHCLayerVsCUDA:
    """Tests comparing CuTe DSL implementation to original CUDA."""
    
    @pytest.mark.parametrize("B,n,C", [(8, 4, 128), (32, 8, 256)])
    def test_matches_cuda_forward(self, B, n, C, device, seed):
        """Test DSL implementation matches original CUDA."""
        from mhc import MHCLayer  # Original CUDA implementation
        
        # Create layers with same weights
        layer_cuda = MHCLayer(hidden_dim=C, expansion_rate=n).to(device)
        layer_dsl = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        
        # Copy weights
        layer_dsl.load_state_dict(layer_cuda.state_dict())
        
        x = torch.randn(B, n, C, device=device)
        
        out_cuda = layer_cuda(x)
        out_dsl = layer_dsl(x)
        
        torch.testing.assert_close(out_dsl, out_cuda, atol=1e-2, rtol=1e-2)
```

### 7.8 Benchmark Suite

```python
"""Performance benchmarks for CuTe DSL kernels."""

import pytest
import torch
import time
from typing import Callable


def benchmark_kernel(
    fn: Callable,
    *args,
    warmup: int = 10,
    iters: int = 100,
    **kwargs
) -> float:
    """Benchmark a kernel function.
    
    Returns:
        Average execution time in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / iters * 1000


class TestRMSNormPerformance:
    """RMSNorm performance benchmarks."""
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("N", [1024, 4096, 16384, 65536])
    @pytest.mark.parametrize("M", [32, 1024, 8192])
    def test_rmsnorm_performance(self, M, N, device, benchmark):
        """Benchmark RMSNorm against CUDA baseline."""
        from mhc.cute_dsl.rmsnorm import rmsnorm as rmsnorm_dsl
        from mhc import rmsnorm as rmsnorm_cuda
        
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        weight = torch.randn(N, device=device, dtype=torch.float32)
        
        time_dsl = benchmark_kernel(rmsnorm_dsl, x, weight)
        time_cuda = benchmark_kernel(rmsnorm_cuda, x, weight)
        
        ratio = time_dsl / time_cuda
        
        # Calculate memory bandwidth
        bytes_transferred = M * N * 2 * 2  # Read input + write output, bf16
        bandwidth_dsl = bytes_transferred / (time_dsl / 1000) / 1e12  # TB/s
        
        print(f"M={M}, N={N}: DSL={time_dsl:.3f}ms, CUDA={time_cuda:.3f}ms, "
              f"ratio={ratio:.2f}x, bandwidth={bandwidth_dsl:.2f} TB/s")
        
        # Performance target: within 10% of CUDA
        assert ratio < 1.1, f"DSL is {ratio:.2f}x slower than CUDA"


# Run benchmarks with: pytest tests/benchmarks/ -v --benchmark
```

### 7.9 Running Tests

```bash
# Run all tests
pytest tests/test_cute_dsl/ -v

# Run specific test file
pytest tests/test_cute_dsl/test_rmsnorm.py -v

# Run with coverage
pytest tests/test_cute_dsl/ --cov=mhc.cute_dsl --cov-report=html

# Run only fast tests (skip slow parametric tests)
pytest tests/test_cute_dsl/ -v -m "not slow"

# Run benchmarks
pytest tests/benchmarks/ -v --benchmark

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/test_cute_dsl/ -v

# Parallel execution
pytest tests/test_cute_dsl/ -v -n auto
```

### 7.10 CI Configuration (`.github/workflows/test.yml`)

```yaml
name: CuTe DSL Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: pip install -e '.[dev]'
      
      - name: Run correctness tests
        run: pytest tests/test_cute_dsl/ -v --tb=short
      
      - name: Run performance regression tests
        run: pytest tests/benchmarks/ -v --benchmark --benchmark-compare
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

## 10. Quack Reference Implementation

The [Quack library](https://github.com/Dao-AILab/quack) from Dao-AILab provides production-quality CuTe DSL kernels that achieve speed-of-light performance. We should leverage their patterns and utilities extensively.

### 10.1 Key Utilities to Adopt from Quack

#### Reduction Utilities (`quack/reduce.py`)

```python
# Block reduction with support for multiple warps per row
@cute.jit
def block_reduce(
    val: cute.Numeric, 
    op: Callable, 
    reduction_buffer: cute.Tensor, 
    init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    """reduction_buffer has shape (num_warps / warp_per_row, warps_per_row)"""
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)

# Row reduction with optional cluster support (for N > 16k)
@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,  # For cluster reduction
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """Unified row reduction supporting warp, block, and cluster levels."""
    ...
```

#### Fast Math Utilities (`quack/fast_math.py`)

```python
from dataclasses import dataclass

@dataclass
class FastDivmod(ParamsBase):
    """Fast integer division for index calculations."""
    divisor: Int32
    multiplier: Uint32
    shift_right: Uint32

    @staticmethod
    def create(divisor: Int32) -> "FastDivmod":
        """Precompute division parameters on host."""
        p = Uint32(31 + find_log2(divisor))
        divisor_u32 = Uint32(divisor)
        multiplier = Uint32(((cutlass.Uint64(1) << p) + divisor_u32 - 1) // divisor_u32)
        shift_right = Uint32(p - 32)
        return FastDivmod(divisor, multiplier, shift_right)

    @cute.jit
    def div(self, dividend: Int32) -> Int32:
        return Int32(umulhi(dividend, self.multiplier) >> self.shift_right) \
               if self.divisor != 1 else dividend

    def divmod(self, dividend: Int32) -> Tuple[Int32, Int32]:
        quotient = self.div(dividend)
        remainder = dividend - quotient * self.divisor
        return quotient, remainder
```

#### Copy Utilities (`quack/copy_utils.py`)

```python
def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    """Create 2D tiled copy for row-major data with vectorized loads."""
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    """Compute predicates for K-dimension bounds checking."""
    ...
```

#### General Utilities (`quack/utils.py`)

```python
@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)

@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *, loc=None, ip=None,
) -> None:
    """Store to another CTA's shared memory via distributed shared memory."""
    ...

@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    """Pack two f32 values into i64 for efficient reduction buffer transfers."""
    ...

@dsl_user_op  
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Unpack i64 back to two f32 values."""
    ...

@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor."""
    ...
```

### 10.2 Quack Kernel Architecture Patterns

#### ReductionBase Class Pattern

```python
class ReductionBase:
    """Base class for reduction kernels (RMSNorm, Softmax, etc.)"""
    
    def __init__(self, dtype: Type[cutlass.Numeric], N: int, stage: int, reduction_dtype=Float32):
        self.dtype = dtype
        self.N = N
        self.stage = stage  # For double-buffering
        self.reduction_dtype = reduction_dtype

    def _threads_per_row(self) -> int:
        """Select threads per row based on reduction dimension."""
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        """Set cluster size for distributed shared memory reduction."""
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [(16*1024, 1), (32*1024, 2), (64*1024, 4), (128*1024, 8)]
        else:
            thresholds = [(32*1024, 1), (64*1024, 2), (128*1024, 4), (256*1024, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    def _get_tiled_copy(self, vecsize: int = 1):
        """Create tiled copy configuration."""
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = copy_utils.tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _allocate_reduction_buffer_and_mbar(self, smem, tv_layout):
        """Allocate reduction buffer and barrier for cluster reduction."""
        ...
```

#### Kernel Launch Pattern with SmemAllocator

```python
@cute.kernel
def kernel(
    self,
    mX: cute.Tensor,
    mW: Optional[cute.Tensor],
    mO: cute.Tensor,
    eps: Float32,
    tiler_mn: cute.Shape,
    tiled_copy: cute.TiledCopy,
    threads_per_row: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    
    # Use SmemAllocator for organized shared memory management
    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(
        mX.element_type, 
        cute.make_ordered_layout(tiler_mn, order=(1, 0)), 
        byte_alignment=16
    )
    reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)
    
    # Get thread-specific copy slice
    thr_copy_X = tiled_copy.get_slice(tidx)
    
    # Partition tensors for this thread
    tXgX = thr_copy_X.partition_S(gX)
    tXsX = thr_copy_X.partition_D(sX)
    
    # Async copy from global to shared memory
    copy(tXgX, tXsX, is_async=True)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    
    # Load from shared to registers
    cute.autovec_copy(tXsX, tXrX)
    x = tXrX.load().to(cute.Float32)
    
    # Perform reduction
    sum_sq_x = row_reduce(
        x * x,
        cute.ReductionOp.ADD,
        threads_per_row,
        reduction_buffer[None, None, 0],
        mbar_ptr,
        init_val=0.0,
    )
    ...
```

### 10.3 Key Paradigms to Adopt

| Pattern | Description | Where to Apply |
|---------|-------------|----------------|
| **SmemAllocator** | Structured shared memory allocation with alignment | All kernels using smem |
| **TiledCopy** | Vectorized loads with thread-value layouts | RMSNorm, Stream Ops |
| **ReductionBase** | Base class with common reduction infrastructure | All reduction kernels |
| **Cluster Reduction** | Distributed smem for large reductions (N > 16k) | RMSNorm with large C |
| **FastDivmod** | Fast integer division for index calculations | Batched kernels |
| **Predicate Tensors** | Efficient bounds checking for uneven dimensions | All kernels |
| **f32x2 Packing** | Pack two f32 values for efficient smem transfers | Online softmax pattern |
| **Async Copy Pipeline** | `cp_async_commit_group` / `wait_group` patterns | Memory-bound kernels |

### 10.4 Memory Hierarchy Strategy (from Quack Blogpost)

For memory-bound kernels, follow the reduction strategy:

| Execution Granularity | Operating Memory | Reduction Strategy |
|----------------------|------------------|-------------------|
| Threads | Registers | `TensorSSA.reduce()` |
| Warps | Registers | `cute.arch.warp_reduction()` |
| Thread Blocks | Shared Memory | `block_reduce()` with smem buffer |
| Thread Block Clusters | Distributed Shared Memory | `cluster_reduce()` with mbarrier |

**Key insight**: Maximize local reduction at higher memory levels, only forward small intermediate results to next level.

### 10.5 Updated Code Organization

```
src/python/mhc/
├── cute_dsl/
│   ├── __init__.py
│   ├── base.py                 # ReductionBase class (from Quack pattern)
│   ├── reduce.py               # block_reduce, row_reduce, cluster_reduce
│   ├── copy_utils.py           # tiled_copy_1d/2d, predicate_k, async copy
│   ├── utils.py                # elem_pointer, store_shared_remote, f32x2 pack
│   ├── fast_math.py            # FastDivmod, clz, umulhi
│   ├── rmsnorm.py              # RMSNorm kernel class
│   ├── sinkhorn.py             # Sinkhorn-Knopp kernels  
│   ├── stream_ops.py           # Stream aggregation/distribution
│   └── mhc_layer.py            # Complete MHC layer
```

---

## 11. References

1. [CUTLASS Python DSL Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
2. [CuTe DSL Introduction](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html)
3. [CuTe DSL Control Flow](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.html)
4. [Framework Integration Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/framework_integration.html)
5. **[Quack: A Quirky Assortment of CuTe Kernels](https://github.com/Dao-AILab/quack)** - Reference implementation
6. **[Getting Memory-bound Kernels to Speed-of-Light](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)** - Quack blogpost on memory hierarchy
7. Original MHC CUDA implementation in this repository

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
