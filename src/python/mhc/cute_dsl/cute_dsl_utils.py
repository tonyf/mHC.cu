"""Core utilities for CuTe DSL kernels - type definitions and base classes.

Adapted from Quack patterns for MHC kernels.
"""

from dataclasses import dataclass, fields
from typing import Type, Optional, Dict, Any
import torch

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Float16, BFloat16, Int32, Int64, Uint32

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    # Provide stubs for type hints
    Float32 = None
    Float16 = None
    BFloat16 = None
    Int32 = None
    Int64 = None
    Uint32 = None


# PyTorch to CuTe dtype mapping
torch2cute_dtype_map: Dict[torch.dtype, Any] = {}
if CUTLASS_AVAILABLE:
    torch2cute_dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
    }

# CuTe to PyTorch dtype mapping
cute2torch_dtype_map: Dict[Any, torch.dtype] = {}
if CUTLASS_AVAILABLE:
    cute2torch_dtype_map = {v: k for k, v in torch2cute_dtype_map.items()}


def get_cute_dtype(torch_dtype: torch.dtype):
    """Get CuTe dtype from PyTorch dtype."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")
    if torch_dtype not in torch2cute_dtype_map:
        raise ValueError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


def get_torch_dtype(cute_dtype) -> torch.dtype:
    """Get PyTorch dtype from CuTe dtype."""
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")
    if cute_dtype not in cute2torch_dtype_map:
        raise ValueError(f"Unsupported dtype: {cute_dtype}")
    return cute2torch_dtype_map[cute_dtype]


@dataclass
class ParamsBase:
    """Base class for kernel parameters that can be passed to device.

    Dataclass fields become kernel arguments. Supports conversion to tuple
    for kernel invocation.
    """

    def to_tuple(self):
        """Convert params to tuple for kernel call."""
        return tuple(getattr(self, f.name) for f in fields(self))


@dataclass
class ArgumentsBase:
    """Base class for kernel arguments including tensors.

    Similar to ParamsBase but for arguments that include tensor pointers.
    """

    def validate(self):
        """Validate arguments before kernel launch."""
        pass


def check_cuda_tensor(tensor: torch.Tensor, name: str = "tensor"):
    """Ensure tensor is on CUDA device."""
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device: {tensor.device}")


def check_contiguous(tensor: torch.Tensor, name: str = "tensor"):
    """Ensure tensor is contiguous."""
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return contiguous tensor, making a copy if necessary."""
    return tensor.contiguous() if not tensor.is_contiguous() else tensor


class KernelRegistry:
    """Registry for compiled kernels with caching."""

    def __init__(self):
        self._cache: Dict[tuple, Any] = {}

    def get_or_compile(self, key: tuple, compile_fn):
        """Get cached kernel or compile new one."""
        if key not in self._cache:
            self._cache[key] = compile_fn()
        return self._cache[key]

    def clear(self):
        """Clear the kernel cache."""
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# Global kernel registry
_kernel_registry = KernelRegistry()


def get_kernel_registry() -> KernelRegistry:
    """Get the global kernel registry."""
    return _kernel_registry
