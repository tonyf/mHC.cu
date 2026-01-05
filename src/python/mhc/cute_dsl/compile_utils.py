"""Compilation utilities for CuTe DSL kernels.

Adapted from Quack patterns for kernel compilation with PyTorch integration.
"""

from typing import Optional, Type, Tuple, Any
import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Float16, BFloat16, Int32

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def make_fake_tensor(
    dtype,
    shape: Tuple,
    alignment_divisor: int = 1,
    dynamic_dims: Optional[Tuple[int, ...]] = None,
):
    """Create fake tensor for kernel compilation.

    Args:
        dtype: CuTe dtype (Float32, BFloat16, etc.) or None
        shape: Tensor shape (can include symbolic dimensions)
        alignment_divisor: For vectorized loads, ensure shape is divisible by this
        dynamic_dims: Indices of dimensions to mark as dynamic

    Returns:
        Fake CuTe tensor suitable for compilation, or None if dtype is None
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

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


def make_symbolic_batch():
    """Create a symbolic batch dimension for shape-agnostic compilation."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")
    return cute.symbolic.make_int_symbol()


def get_cuda_stream():
    """Get current CUDA stream from PyTorch."""
    try:
        import cuda.bindings.driver as cuda

        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    except ImportError:
        # Fallback for older CUDA bindings
        return torch.cuda.current_stream().cuda_stream


def make_fake_stream():
    """Create a fake stream for compilation with TVM FFI."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def compile_kernel(kernel_class, *fake_args, **compile_options):
    """Compile a kernel class with fake arguments.

    Args:
        kernel_class: Kernel class instance with __call__ method
        *fake_args: Fake tensor arguments for compilation
        **compile_options: Additional options for cute.compile

    Returns:
        Compiled kernel callable
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    default_options = {"options": "--enable-tvm-ffi"}
    default_options.update(compile_options)

    return cute.compile(kernel_class, *fake_args, **default_options)


def get_vectorization_size(dtype, N: int) -> int:
    """Calculate optimal vectorization size for given dtype and dimension.

    Args:
        dtype: CuTe dtype
        N: Dimension to vectorize

    Returns:
        Optimal vector size (1, 2, 4, or 8)
    """
    if not CUTLASS_AVAILABLE:
        return 1

    # Maximum 128 bits per load
    max_bits = 128
    dtype_bits = dtype.width

    max_vec = max_bits // dtype_bits

    # Find largest divisor
    for vec in [8, 4, 2, 1]:
        if vec <= max_vec and N % vec == 0:
            return vec

    return 1


def get_num_warps(block_size: int) -> int:
    """Calculate number of warps for given block size."""
    WARP_SIZE = 32
    return block_size // WARP_SIZE


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b
