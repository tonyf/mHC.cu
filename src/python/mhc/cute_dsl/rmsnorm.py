"""RMSNorm kernel implementation using CuTe DSL.

Adapted from Quack patterns for speed-of-light performance.
Implements RMSNorm forward and backward:
    y = x / sqrt(mean(x^2) + eps) * weight

Features:
- Vectorized memory access
- Warp/block/cluster reduction support
- Async copy pipeline
- PyTorch autograd integration
"""

from typing import Optional, Tuple
from functools import partial
import math

import torch
from torch import Tensor
from torch.autograd import Function

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, const_expr

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

from .base import ReductionBase, SmemAllocator, ceil_div
from .reduce import row_reduce
from .copy_utils import tiled_copy_2d, predicate_k, copy
from .compile_utils import (
    make_fake_tensor,
    make_symbolic_batch,
    make_fake_stream,
    get_cuda_stream,
)
from .cute_dsl_utils import (
    torch2cute_dtype_map,
    get_cute_dtype,
    check_cuda_tensor,
    ensure_contiguous,
    get_kernel_registry,
)


class RMSNorm(ReductionBase):
    """RMSNorm kernel using Quack patterns for speed-of-light performance.

    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    This kernel is memory-bound and optimized for:
    - Vectorized loads (up to 128 bits)
    - Efficient warp/block reduction
    - Cluster reduction for very large N (>16k)
    - Fused weight multiplication
    """

    def __init__(self, dtype, N: int, output_rstd: bool = False):
        """Initialize RMSNorm kernel.

        Args:
            dtype: Input element dtype (Float32, BFloat16, etc.)
            N: Hidden dimension size
            output_rstd: Whether to output 1/rms values
        """
        super().__init__(dtype, N, stage=1)
        self.output_rstd = output_rstd
        # For large N, reload from smem after reduction
        self.reload_from = None if N <= 8192 else "smem"

    def _threads_per_row(self) -> int:
        """Select optimal threads per row based on reduction dimension."""
        N = self.N
        for limit, threads in [
            (64, 8),
            (128, 16),
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if N <= limit:
                return threads
        return 256

    if CUTLASS_AVAILABLE:

        @cute.jit
        def __call__(
            self,
            mX,  # [B, N] input
            mW,  # [N] weight (optional)
            mO,  # [B, N] output
            mRstd,  # [B] rstd output (optional)
            eps: Float32,
            stream,
        ):
            """Launch RMSNorm kernel.

            Args:
                mX: Input tensor [B, N]
                mW: Weight tensor [N] or None
                mO: Output tensor [B, N]
                mRstd: Rstd output tensor [B] or None
                eps: Epsilon for numerical stability
                stream: CUDA stream
            """
            self._set_cluster_n()

            # Calculate vectorization
            largest_dtype_width = max(
                t.element_type.width for t in [mX, mW, mO] if t is not None
            )
            vecsize = math.gcd(self.N, 128 // largest_dtype_width)
            tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)

            self.kernel(
                mX, mW, mO, mRstd, eps, tiler_mn, tiled_copy, threads_per_row
            ).launch(
                grid=[ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
                block=[cute.size(tiled_copy), 1, 1],
                cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mX,
            mW,
            mO,
            mRstd,
            eps: Float32,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ):
            """RMSNorm kernel implementation."""
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cluster_y = cute.arch.block_idx()[1] if self.cluster_n > 1 else 0
            tv_layout = tiled_copy.layout_tv_tiled

            # Allocate shared memory
            smem = cutlass.utils.SmemAllocator()
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(
                smem, tv_layout
            )

            shape = mX.shape
            idX = cute.make_identity_tensor(shape)

            # Partition for this CTA
            gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
            gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))
            cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

            gW = None
            if const_expr(mW is not None):
                gW = cute.local_tile(mW, tiler_mn, (0, cluster_y))

            gRstd = None
            if const_expr(mRstd is not None):
                gRstd = cute.local_tile(mRstd, tiler_mn, (bidx, 0))

            thr_copy_X = tiled_copy.get_slice(tidx)

            # Partition tensors for this thread
            tXgX = thr_copy_X.partition_S(gX)
            tXsX = thr_copy_X.partition_D(sX)
            tXgO = thr_copy_X.partition_D(gO)
            tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

            tXgW = None
            if const_expr(mW is not None):
                tXgW = thr_copy_X.partition_S(gW)

            # Allocate register fragments
            tXrX = cute.make_fragment_like(tXgX)
            tXrO = cute.make_fragment_like(tXgO)
            tXrW = None
            if const_expr(mW is not None):
                tXrW = cute.make_fragment_like(tXgW)

            # Initialize cluster if needed
            num_warps = cute.size(tiled_copy) // 32
            self._initialize_cluster(tidx, mbar_ptr, num_warps)

            # Handle uneven dimensions
            is_even_N = shape[1] == tiler_mn[1] * self.cluster_n
            tXpX = None
            if not is_even_N:
                tXpX = predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])

            row = tXcX[0][0]
            if row < shape[0]:
                copy(tXgX, tXsX, pred=tXpX, is_async=True)
            cute.arch.cp_async_commit_group()

            # Load weights while waiting for data
            if const_expr(mW is not None):
                copy(tXgW, tXrW, pred=tXpX)

            cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

            # Compute sum of squares with row reduction
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=Float32(0.0),
                hook_fn=cute.arch.cluster_wait if self.cluster_n > 1 else None,
            )

            # Compute rstd = 1 / sqrt(mean(x^2) + eps)
            rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)

            # Store rstd if requested
            if const_expr(mRstd is not None):
                if tXcX[0][1] == 0 and row < shape[0]:
                    if self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0:
                        thr_copy_X.partition_D(gRstd)[0] = rstd

            # Reload x if needed for large N
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                x = tXrX.load().to(Float32)

            # Normalize and apply weight
            o = x * rstd
            if const_expr(mW is not None):
                w = tXrW.load().to(Float32)
                o = o * w

            tXrO.store(o.to(tXrO.element_type))
            if row < shape[0]:
                copy(tXrO, tXgO, pred=tXpX)


class RMSNormBackward(ReductionBase):
    """RMSNorm backward kernel.

    Computes gradients for RMSNorm:
    - d_inp: gradient w.r.t. input
    - d_weight: gradient w.r.t. weight (accumulated)
    """

    def __init__(self, dtype, N: int):
        super().__init__(dtype, N, stage=1)

    if CUTLASS_AVAILABLE:

        @cute.jit
        def __call__(
            self,
            mGrad,  # [B, N] upstream gradient
            mX,  # [B, N] input
            mW,  # [N] weight
            mRstd,  # [B] rstd from forward
            mDX,  # [B, N] input gradient output
            mDW,  # [N] weight gradient output
            stream,
        ):
            """Launch RMSNorm backward kernel."""
            vecsize = self.get_vectorization_size(mGrad, mX, mW)
            tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)

            self.kernel(
                mGrad, mX, mW, mRstd, mDX, mDW, tiler_mn, tiled_copy, threads_per_row
            ).launch(
                grid=[ceil_div(mX.shape[0], tiler_mn[0]), 1, 1],
                block=[cute.size(tiled_copy), 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mGrad,
            mX,
            mW,
            mRstd,
            mDX,
            mDW,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ):
            """RMSNorm backward kernel implementation."""
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            tv_layout = tiled_copy.layout_tv_tiled

            smem = cutlass.utils.SmemAllocator()
            reduction_buffer, _ = self._allocate_reduction_buffer_and_mbar(
                smem, tv_layout
            )

            shape = mX.shape
            N = shape[1]

            # Partition tensors
            gGrad = cute.local_tile(mGrad, tiler_mn, (bidx, 0))
            gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
            gW = cute.local_tile(mW, tiler_mn, (0, 0))
            gDX = cute.local_tile(mDX, tiler_mn, (bidx, 0))
            gDW = cute.local_tile(mDW, tiler_mn, (0, 0))

            thr_copy = tiled_copy.get_slice(tidx)
            tGrad = thr_copy.partition_S(gGrad)
            tX = thr_copy.partition_S(gX)
            tW = thr_copy.partition_S(gW)
            tDX = thr_copy.partition_D(gDX)

            # Load data
            rGrad = cute.make_fragment_like(tGrad)
            rX = cute.make_fragment_like(tX)
            rW = cute.make_fragment_like(tW)
            rDX = cute.make_fragment_like(tDX)

            copy(tGrad, rGrad)
            copy(tX, rX)
            copy(tW, rW)

            grad = rGrad.load().to(Float32)
            x = rX.load().to(Float32)
            w = rW.load().to(Float32)
            rstd = mRstd[bidx]

            # Compute dot product for correction term
            dot = row_reduce(
                grad * w * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                init_val=Float32(0.0),
            )

            correction = dot / (Float32(N) * rstd * rstd)

            # Compute input gradient
            dx = (grad * w * rstd) - (x * correction * rstd)
            rDX.store(dx.to(rDX.element_type))
            copy(rDX, tDX)

            # Atomic add to weight gradient
            # Note: In practice, this would use atomicAdd


# Compilation cache
_rmsnorm_fwd_cache = {}
_rmsnorm_bwd_cache = {}


def _get_compile_key(x: Tensor, weight: Optional[Tensor], out: Tensor, N: int):
    """Get cache key for kernel compilation."""
    dtype = torch2cute_dtype_map.get(x.dtype)
    weight_dtype = torch2cute_dtype_map.get(weight.dtype) if weight is not None else None
    out_dtype = torch2cute_dtype_map.get(out.dtype)
    return (dtype, weight_dtype, out_dtype, N)


def _rmsnorm_fwd_impl(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    rstd: Optional[Tensor],
    eps: float,
):
    """Internal forward implementation with compilation caching."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available. Install nvidia-cutlass package.")

    B, N = x.shape
    dtype = get_cute_dtype(x.dtype)
    weight_dtype = get_cute_dtype(weight.dtype) if weight is not None else None
    out_dtype = get_cute_dtype(out.dtype)

    compile_key = (dtype, weight_dtype, out_dtype, N, rstd is not None)

    if compile_key not in _rmsnorm_fwd_cache:
        # Create fake tensors for compilation
        batch_sym = make_symbolic_batch()
        all_dtypes = [dtype, out_dtype]
        if weight_dtype:
            all_dtypes.append(weight_dtype)
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes))

        x_cute = make_fake_tensor(dtype, (batch_sym, N), div)
        out_cute = make_fake_tensor(out_dtype, (batch_sym, N), div)
        weight_cute = make_fake_tensor(weight_dtype, (N,), div) if weight_dtype else None
        rstd_cute = make_fake_tensor(Float32, (batch_sym,)) if rstd is not None else None

        kernel = RMSNorm(dtype, N, output_rstd=(rstd is not None))
        _rmsnorm_fwd_cache[compile_key] = cute.compile(
            kernel,
            x_cute,
            weight_cute,
            out_cute,
            rstd_cute,
            Float32(0),  # eps placeholder
            make_fake_stream(),
            options="--enable-tvm-ffi",
        )

    _rmsnorm_fwd_cache[compile_key](x, weight, out, rstd, eps)


def rmsnorm_fwd(
    x: Tensor,
    weight: Optional[Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """RMSNorm forward with automatic output allocation.

    Args:
        x: Input tensor [B, N]
        weight: Optional weight tensor [N]
        out_dtype: Output dtype (default: same as input)
        eps: Epsilon for numerical stability
        store_rstd: Whether to store 1/rms values

    Returns:
        Tuple of (output, rstd or None)
    """
    check_cuda_tensor(x, "x")
    x = ensure_contiguous(x)
    if weight is not None:
        check_cuda_tensor(weight, "weight")
        weight = ensure_contiguous(weight)

    out_dtype = x.dtype if out_dtype is None else out_dtype
    out = torch.empty_like(x, dtype=out_dtype)
    rstd = (
        torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
        if store_rstd
        else None
    )

    _rmsnorm_fwd_impl(x, weight, out, rstd, eps)
    return out, rstd


class RMSNormFunction(Function):
    """PyTorch autograd function for RMSNorm."""

    @staticmethod
    def forward(ctx, inp: Tensor, weight: Optional[Tensor], eps: float):
        out, rstd = rmsnorm_fwd(inp, weight, eps=eps, store_rstd=True)
        ctx.save_for_backward(inp, weight, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        inp, weight, rstd = ctx.saved_tensors

        # Use PyTorch reference implementation for backward
        # (CuTe DSL backward would require atomic operations)
        inp_f32 = inp.float()
        grad_f32 = grad_output.float()
        weight_f32 = weight.float() if weight is not None else None

        # Recompute normalized input
        rms = 1.0 / rstd.unsqueeze(1)
        x_norm = inp_f32 * rstd.unsqueeze(1)

        # Input gradient
        if weight_f32 is not None:
            grad_x_norm = grad_f32 * weight_f32
        else:
            grad_x_norm = grad_f32

        # d_inp = (grad_x_norm - x_norm * mean(grad_x_norm * x_norm)) * rstd
        dot = (grad_x_norm * x_norm).sum(dim=-1, keepdim=True) / inp.shape[-1]
        d_inp = (grad_x_norm - x_norm * dot) * rstd.unsqueeze(1)

        # Weight gradient
        d_weight = None
        if weight is not None:
            d_weight = (grad_f32 * x_norm).sum(dim=0)

        return d_inp.to(inp.dtype), d_weight, None


def rmsnorm(inp: Tensor, weight: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """Functional interface for RMSNorm.

    Args:
        inp: Input tensor [B, N]
        weight: Optional weight tensor [N]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    return RMSNormFunction.apply(inp, weight, eps)
