"""Reduction operations for CuTe DSL kernels.

Adapted from Quack's reduce.py for MHC kernels. Provides:
- Warp-level reductions
- Block-level reductions via shared memory
- Cluster-level reductions via distributed shared memory
- Unified row_reduce function supporting all levels
"""

import operator
from typing import Callable, Optional

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Float32, const_expr

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

from .utils import elem_pointer, store_shared_remote

# Warp size constant
WARP_SIZE = 32


def warp_reduce_sum(val: Float32) -> Float32:
    """Butterfly reduction within a warp for sum.

    Args:
        val: Per-thread value

    Returns:
        Sum across warp (all lanes get same result)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _warp_reduce_sum_impl(val):
        for offset in cutlass.range_constexpr(5):  # log2(32) = 5
            mask = 1 << (4 - offset)
            val = val + cute.arch.shfl_xor(val, mask)
        return val

    return _warp_reduce_sum_impl(val)


def warp_reduce_max(val: Float32) -> Float32:
    """Butterfly reduction within a warp for max.

    Args:
        val: Per-thread value

    Returns:
        Max across warp (all lanes get same result)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _warp_reduce_max_impl(val):
        for offset in cutlass.range_constexpr(5):  # log2(32) = 5
            mask = 1 << (4 - offset)
            other = cute.arch.shfl_xor(val, mask)
            val = cute.arch.fmax(val, other)
        return val

    return _warp_reduce_max_impl(val)


def block_reduce(
    val,
    op: Callable,
    reduction_buffer,
    init_val=0.0,
):
    """Block reduction via shared memory.

    Args:
        val: Per-thread value to reduce
        op: Reduction operator (add, max, etc.)
        reduction_buffer: Shared memory buffer shape (num_warps/warps_per_row, warps_per_row)
        init_val: Initial value for reduction

    Returns:
        Reduced value (valid in lane 0 of each row's first warp)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _block_reduce_impl(val, op, reduction_buffer, init_val):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        warps_per_row = cute.size(reduction_buffer.shape[1])
        row_idx = warp_idx // warps_per_row
        col_idx = warp_idx % warps_per_row

        if lane_idx == 0:
            reduction_buffer[row_idx, col_idx] = val
        cute.arch.barrier()

        block_reduce_val = init_val
        if lane_idx < warps_per_row:
            block_reduce_val = reduction_buffer[row_idx, lane_idx]
        return cute.arch.warp_reduction(block_reduce_val, op)

    return _block_reduce_impl(val, op, reduction_buffer, init_val)


def cluster_reduce(
    val,
    op: Callable,
    reduction_buffer,
    mbar_ptr,
    init_val=0.0,
    phase: Optional[Int32] = None,
):
    """Cluster reduction via distributed shared memory.

    For very large reductions (N > 16k) that span multiple CTAs.

    Args:
        val: Per-thread value to reduce
        op: Reduction operator
        reduction_buffer: Shape (num_warps/warps_per_row, (warps_per_row, cluster_n))
        mbar_ptr: Memory barrier pointer
        init_val: Initial value
        phase: Barrier phase

    Returns:
        Reduced value
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _cluster_reduce_impl(val, op, reduction_buffer, mbar_ptr, init_val, phase):
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
        row_idx = warp_idx // warps_per_row
        col_idx = warp_idx % warps_per_row

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
                elem_pointer(
                    reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))
                ),
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

    return _cluster_reduce_impl(val, op, reduction_buffer, mbar_ptr, init_val, phase)


def row_reduce(
    x,
    op,
    threads_per_row,
    reduction_buffer=None,
    mbar_ptr=None,
    phase: Optional[Int32] = None,
    init_val=0.0,
    hook_fn: Optional[Callable] = None,
):
    """Unified row reduction supporting warp, block, and cluster levels.

    This is the main reduction entry point. It automatically handles:
    - Thread-level reduction via TensorSSA.reduce()
    - Warp-level reduction via shuffle operations
    - Block-level reduction via shared memory (if reduction_buffer provided)
    - Cluster-level reduction via distributed smem (if mbar_ptr provided)

    Args:
        x: Input tensor or scalar
        op: Reduction operation (cute.ReductionOp.ADD, MAX, MIN, MUL)
        threads_per_row: Number of threads participating per row
        reduction_buffer: Shared memory for block/cluster reduction
        mbar_ptr: Memory barrier for cluster reduction
        phase: Barrier phase
        init_val: Initial value
        hook_fn: Hook function called between warp and block reduction

    Returns:
        Reduced value
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _row_reduce_impl(
        x, op, threads_per_row, reduction_buffer, mbar_ptr, phase, init_val, hook_fn
    ):
        # Thread-level reduction
        if const_expr(isinstance(x, cute.TensorSSA)):
            val = x.reduce(op, init_val=init_val, reduction_profile=0)
        else:
            val = x

        # Map reduction op to warp operator
        warp_op = {
            cute.ReductionOp.ADD: operator.add,
            cute.ReductionOp.MAX: cute.arch.fmax
            if const_expr(hasattr(x, "dtype") and x.dtype == Float32)
            else max,
            cute.ReductionOp.MIN: min,
            cute.ReductionOp.MUL: operator.mul,
        }[op]

        # Warp-level reduction
        val = cute.arch.warp_reduction(
            val, warp_op, threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE)
        )

        # Optional hook between warp and block reduction
        if const_expr(hook_fn is not None):
            hook_fn()

        # Block/cluster-level reduction if needed
        if const_expr(reduction_buffer is not None):
            warps_per_row, cluster_n = reduction_buffer.shape[1]
            if const_expr(warps_per_row > 1 or cluster_n > 1):
                if const_expr(mbar_ptr is None):
                    val = block_reduce(val, warp_op, reduction_buffer, init_val=init_val)
                else:
                    val = cluster_reduce(
                        val,
                        warp_op,
                        reduction_buffer,
                        mbar_ptr,
                        phase=phase,
                        init_val=init_val,
                    )
        return val

    return _row_reduce_impl(
        x, op, threads_per_row, reduction_buffer, mbar_ptr, phase, init_val, hook_fn
    )


def sum_reduce(x, threads_per_row, reduction_buffer=None, mbar_ptr=None):
    """Convenience function for sum reduction.

    Args:
        x: Input tensor
        threads_per_row: Threads per row
        reduction_buffer: Optional shared memory buffer
        mbar_ptr: Optional cluster barrier

    Returns:
        Sum of elements
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return row_reduce(
        x,
        cute.ReductionOp.ADD,
        threads_per_row,
        reduction_buffer=reduction_buffer,
        mbar_ptr=mbar_ptr,
        init_val=Float32(0.0),
    )


def max_reduce(x, threads_per_row, reduction_buffer=None, mbar_ptr=None):
    """Convenience function for max reduction.

    Args:
        x: Input tensor
        threads_per_row: Threads per row
        reduction_buffer: Optional shared memory buffer
        mbar_ptr: Optional cluster barrier

    Returns:
        Maximum element
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return row_reduce(
        x,
        cute.ReductionOp.MAX,
        threads_per_row,
        reduction_buffer=reduction_buffer,
        mbar_ptr=mbar_ptr,
        init_val=Float32(float("-inf")),
    )
