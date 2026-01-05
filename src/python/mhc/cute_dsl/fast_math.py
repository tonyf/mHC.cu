"""Fast math utilities for CuTe DSL kernels.

Adapted from Quack's fast_math.py for MHC kernels. Provides:
- Fast integer division (FastDivmod)
- Count leading zeros
- Fast math operations using PTX intrinsics
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Uint32, Int64, Float32
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass._mlir.dialects import llvm

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False
    Int32 = None
    Uint32 = None
    Int64 = None
    Float32 = None

    def dsl_user_op(fn):
        return fn


from .cute_dsl_utils import ParamsBase


def clz(x):
    """Count leading zeros in 32-bit integer.

    Args:
        x: Input integer

    Returns:
        Number of leading zeros (0-32)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _clz_impl(x):
        res = Int32(32)
        done = False
        for i in cutlass.range(32):
            if ((1 << (31 - i)) & x) and not done:
                res = Int32(i)
                done = True
        return res

    return _clz_impl(x)


def find_log2(x):
    """Find log2 rounded up.

    Args:
        x: Input integer (must be > 0)

    Returns:
        ceil(log2(x))
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _find_log2_impl(x):
        a = Int32(31 - clz(x))
        return a + ((x & (x - 1)) != 0)

    return _find_log2_impl(x)


@dsl_user_op
def umulhi(a, b, *, loc=None, ip=None):
    """Unsigned multiply high - returns high 32 bits of 64-bit product.

    This is a PTX instruction for fast division.

    Args:
        a: First operand
        b: Second operand

    Returns:
        High 32 bits of a*b
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "mul.hi.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dataclass
class FastDivmod(ParamsBase):
    """Fast integer division using multiply-shift technique.

    Precompute division parameters on host, then use fast multiply-shift on device.
    This avoids expensive integer division instructions.

    Usage:
        divmod_helper = FastDivmod.create(divisor)
        quotient = divmod_helper.div(dividend)
        quotient, remainder = divmod_helper.divmod(dividend)
    """

    divisor: int
    multiplier: int
    shift_right: int

    @staticmethod
    def create(divisor: int) -> "FastDivmod":
        """Precompute fast divmod parameters (call on host).

        Args:
            divisor: The divisor (must be > 0)

        Returns:
            FastDivmod instance with precomputed parameters
        """
        if divisor <= 0:
            raise ValueError(f"divisor must be positive, got {divisor}")

        if divisor == 1:
            return FastDivmod(
                divisor=1,
                multiplier=1,
                shift_right=0,
            )

        # Find log2(divisor) rounded up
        log2_div = (divisor - 1).bit_length()
        p = 31 + log2_div

        # Compute multiplier: ceil((2^p) / divisor)
        multiplier = ((1 << p) + divisor - 1) // divisor
        shift_right = p - 32

        return FastDivmod(
            divisor=divisor,
            multiplier=multiplier,
            shift_right=shift_right,
        )

    def div(self, dividend):
        """Fast division on device.

        Args:
            dividend: The dividend

        Returns:
            dividend // divisor
        """
        if not CUTLASS_AVAILABLE:
            raise RuntimeError("CUTLASS not available")

        @cute.jit
        def _div_impl(self_divisor, self_multiplier, self_shift_right, dividend):
            if self_divisor != 1:
                return Int32(umulhi(dividend, self_multiplier) >> self_shift_right)
            else:
                return dividend

        return _div_impl(
            self.divisor, self.multiplier, self.shift_right, dividend
        )

    def divmod(self, dividend):
        """Fast divmod on device.

        Args:
            dividend: The dividend

        Returns:
            Tuple of (quotient, remainder)
        """
        if not CUTLASS_AVAILABLE:
            raise RuntimeError("CUTLASS not available")

        @cute.jit
        def _divmod_impl(self_divisor, self_multiplier, self_shift_right, dividend):
            if self_divisor != 1:
                quotient = Int32(umulhi(dividend, self_multiplier) >> self_shift_right)
            else:
                quotient = dividend
            remainder = dividend - quotient * self_divisor
            return quotient, remainder

        return _divmod_impl(
            self.divisor, self.multiplier, self.shift_right, dividend
        )


# PTX intrinsic wrappers
@dsl_user_op
def fast_expf(x, *, loc=None, ip=None):
    """Fast exponential using PTX __expf.

    Args:
        x: Input value

    Returns:
        exp(x) with reduced precision
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",  # exp2(x) approximation
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fast_rcpf(x, *, loc=None, ip=None):
    """Fast reciprocal using PTX __frcp_rn.

    Args:
        x: Input value

    Returns:
        1/x with round-to-nearest
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fast_rsqrtf(x, *, loc=None, ip=None):
    """Fast reciprocal square root using PTX.

    Args:
        x: Input value

    Returns:
        1/sqrt(x)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fast_sqrtf(x, *, loc=None, ip=None):
    """Fast square root using PTX.

    Args:
        x: Input value

    Returns:
        sqrt(x)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "sqrt.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def fast_sigmoid(x):
    """Fast sigmoid function.

    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input value

    Returns:
        sigmoid(x)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _fast_sigmoid_impl(x):
        exp_neg_x = cute.math.exp(-x, fastmath=True)
        return fast_rcpf(Float32(1.0) + exp_neg_x)

    return _fast_sigmoid_impl(x)


def fast_gelu(x):
    """Fast GELU activation.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Args:
        x: Input value

    Returns:
        GELU(x)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _fast_gelu_impl(x):
        # Constants for GELU approximation
        SQRT_2_OVER_PI = Float32(0.7978845608028654)
        COEFF = Float32(0.044715)

        inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
        # tanh approximation using exp
        exp_2x = cute.math.exp(Float32(2.0) * inner, fastmath=True)
        tanh_approx = (exp_2x - Float32(1.0)) / (exp_2x + Float32(1.0))
        return Float32(0.5) * x * (Float32(1.0) + tanh_approx)

    return _fast_gelu_impl(x)
