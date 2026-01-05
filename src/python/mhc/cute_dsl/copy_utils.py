"""Copy and memory utilities for CuTe DSL kernels.

Adapted from Quack's copy_utils.py for MHC kernels. Provides:
- Tiled copy patterns (1D and 2D)
- Predicate computation for bounds checking
- Async copy utilities
- Register load utilities
"""

from typing import Optional, Type

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Boolean, const_expr
    from cutlass.cute.nvgpu import cpasync

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def load_s2r(src, *, loc=None, ip=None):
    """Load from shared memory to registers.

    Args:
        src: Source tensor in shared memory

    Returns:
        Register tensor with loaded values
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _load_s2r_impl(src):
        dst = cute.make_fragment_like(src, src.element_type)
        cute.autovec_copy(src, dst)
        return dst

    return _load_s2r_impl(src)


def tiled_copy_1d(
    dtype,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
):
    """Create 1D tiled copy pattern.

    Args:
        dtype: Element dtype
        num_threads: Total number of threads
        num_copy_elems: Elements per copy operation (for vectorization)
        is_async: Whether to use async copy

    Returns:
        TiledCopy object
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype,
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
):
    """Create 2D tiled copy for row-major data with vectorized loads.

    This is the primary copy pattern for reduction kernels where each
    row is processed by threads_per_row threads.

    Args:
        dtype: Element dtype
        threads_per_row: Number of threads processing each row
        num_threads: Total number of threads
        num_copy_elems: Elements per copy (vectorization)
        is_async: Whether to use async copy

    Returns:
        TiledCopy object
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)

    assert num_threads % threads_per_row == 0, (
        f"num_threads ({num_threads}) must be divisible by "
        f"threads_per_row ({threads_per_row})"
    )

    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),  # Row-major thread assignment
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def predicate_k(tAcA, limit: Int32):
    """Compute predicates for K-dimension bounds checking.

    Used for handling non-divisible dimensions where some threads
    may be out of bounds.

    Args:
        tAcA: Coordinate tensor
        limit: Upper bound for valid coordinates

    Returns:
        Predicate tensor
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _predicate_k_impl(tAcA, limit):
        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    cute.size(tAcA, mode=[0, 1]),
                    cute.size(tAcA, mode=[1]),
                    cute.size(tAcA, mode=[2]),
                ),
                stride=(cute.size(tAcA, mode=[2]), 0, 1),
            ),
            Boolean,
        )
        for rest_v in cutlass.range_constexpr(tApA.shape[0]):
            for rest_k in cutlass.range_constexpr(tApA.shape[2]):
                tApA[rest_v, 0, rest_k] = cute.elem_less(
                    tAcA[(0, rest_v), 0, rest_k][1], limit
                )
        return tApA

    return _predicate_k_impl(tAcA, limit)


def copy(
    src,
    dst,
    *,
    pred=None,
    is_async: bool = False,
    **kwargs,
):
    """Generic copy with optional predication and async support.

    Args:
        src: Source tensor
        dst: Destination tensor
        pred: Optional predicate tensor
        is_async: Whether to use async copy
        **kwargs: Additional arguments for cute.copy
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _copy_impl(src, dst, pred, is_async):
        num_copy_elems = src.shape[0][0]
        num_copy_bits = min(128, num_copy_elems * src.element_type.width)
        copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
        copy_atom = cute.make_copy_atom(
            copy_op, src.element_type, num_bits_per_copy=num_copy_bits
        )
        cute.copy(copy_atom, src, dst, pred=pred)

    _copy_impl(src, dst, pred, is_async)


def copy_global_to_shared_async(src, dst, pred=None):
    """Async copy from global to shared memory.

    Args:
        src: Source tensor in global memory
        dst: Destination tensor in shared memory
        pred: Optional predicate tensor
    """
    copy(src, dst, pred=pred, is_async=True)


def copy_shared_to_register(src, dst):
    """Copy from shared memory to register.

    Args:
        src: Source tensor in shared memory
        dst: Destination tensor in registers
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    cute.autovec_copy(src, dst)


def copy_register_to_global(src, dst, pred=None):
    """Copy from register to global memory.

    Args:
        src: Source tensor in registers
        dst: Destination tensor in global memory
        pred: Optional predicate tensor
    """
    copy(src, dst, pred=pred, is_async=False)


def async_copy_commit():
    """Commit async copy group."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    cute.arch.cp_async_commit_group()


def async_copy_wait(count: int = 0):
    """Wait for async copy groups to complete.

    Args:
        count: Number of groups to wait for (0 = wait for all)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    cute.arch.cp_async_wait_group(count)


def make_identity_tensor(shape):
    """Create identity tensor for coordinate calculations.

    Args:
        shape: Tensor shape

    Returns:
        Identity tensor where each element is its coordinate
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return cute.make_identity_tensor(shape)
