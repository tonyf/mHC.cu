"""Core utilities for CuTe DSL kernels.

Adapted from Quack's utils.py for MHC kernels. Provides:
- Element pointer operations
- Distributed shared memory operations
- f32x2 packing for efficient transfers
- Out-of-bounds filling utilities
"""

from functools import partial
from typing import Optional, Tuple

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, Int64, const_expr
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass._mlir.dialects import llvm, nvvm, vector

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

    # Provide stub decorators for when CUTLASS is not available
    def dsl_user_op(fn):
        return fn


if CUTLASS_AVAILABLE:
    # Packed f32 operations with proper rounding
    fma_packed_f32x2 = partial(
        cute.arch.fma_packed_f32x2, rnd=nvvm.RoundingModeKind.RN
    )
    mul_packed_f32x2 = partial(
        cute.arch.mul_packed_f32x2, rnd=nvvm.RoundingModeKind.RN
    )
    add_packed_f32x2 = partial(
        cute.arch.add_packed_f32x2, rnd=nvvm.RoundingModeKind.RN
    )


@dsl_user_op
def elem_pointer(x, coord, *, loc=None, ip=None):
    """Get pointer to element at coordinate in tensor.

    Args:
        x: CuTe tensor
        coord: Coordinate tuple

    Returns:
        Pointer to element at coordinate
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(smem_ptr, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None):
    """Map smem pointer to address at another CTA rank in cluster.

    Used for distributed shared memory operations.

    Args:
        smem_ptr: Shared memory pointer
        peer_cta_rank_in_cluster: Target CTA rank

    Returns:
        Remapped pointer as Int32
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val,
    smem_ptr,
    mbar_ptr,
    peer_cta_rank_in_cluster,
    *,
    loc=None,
    ip=None,
):
    """Store to another CTA's shared memory via distributed shared memory.

    Args:
        val: Value to store (Float32 or Int32)
        smem_ptr: Target shared memory pointer
        mbar_ptr: Memory barrier pointer
        peer_cta_rank_in_cluster: Target CTA rank
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()

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
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None):
    """Pack two f32 values into i64 for efficient smem transfers.

    Args:
        a: First f32 value
        b: Second f32 value

    Returns:
        Packed Int64 value
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    return cutlass.Int64(
        vector.extract(
            vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )


@dsl_user_op
def i64_to_f32x2(c, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Unpack i64 back to two f32 values.

    Args:
        c: Packed Int64 value

    Returns:
        Tuple of two Float32 values
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    vec_i64x1 = vector.from_elements(
        T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip
    )
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(
            vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip
        )
    )
    res1 = Float32(
        vector.extract(
            vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip
        )
    )
    return res0, res1


@dsl_user_op
def atomic_add_i32(a, gmem_ptr, *, loc=None, ip=None):
    """Atomic add for int32.

    Args:
        a: Value to add
        gmem_ptr: Global memory pointer

    Returns:
        Previous value at pointer
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return nvvm.atomicrmw(
        res=T.i32(),
        op=nvvm.AtomicOpKind.ADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Int32(a).ir_value(),
    )


@dsl_user_op
def atomic_add_f32(a, gmem_ptr, *, loc=None, ip=None):
    """Atomic add for float32.

    Args:
        a: Value to add
        gmem_ptr: Global memory pointer

    Returns:
        Previous value at pointer
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return nvvm.atomicrmw(
        res=T.f32(),
        op=nvvm.AtomicOpKind.ADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(a).ir_value(),
    )


def fill_oob(tXsX, tXpX: Optional, fill_value) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor partition
        tXpX: Predicate tensor (or None)
        fill_value: Value to fill OOB positions
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    @cute.jit
    def _fill_oob_impl(tXsX, tXpX, fill_value):
        tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
        tXrX_fill.fill(fill_value)
        for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
            for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
                if const_expr(tXpX is not None):
                    if not tXpX[rest_v, 0, rest_k]:
                        cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
                else:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])

    _fill_oob_impl(tXsX, tXpX, fill_value)


def fast_exp(x: Float32) -> Float32:
    """Fast exponential using PTX intrinsic.

    Args:
        x: Input value

    Returns:
        exp(x) using fast approximation
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    # Use cute.math.exp with fastmath for PTX __expf
    return cute.math.exp(x, fastmath=True)


def fast_sigmoid(x: Float32) -> Float32:
    """Fast sigmoid using fast reciprocal.

    sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x: Input value

    Returns:
        sigmoid(x)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    exp_neg_x = cute.math.exp(-x, fastmath=True)
    return cute.math.rsqrt(
        (Float32(1.0) + exp_neg_x) * (Float32(1.0) + exp_neg_x), fastmath=True
    ) * (Float32(1.0) + exp_neg_x)


def fast_rsqrt(x: Float32) -> Float32:
    """Fast reciprocal square root.

    Args:
        x: Input value

    Returns:
        1 / sqrt(x)
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    return cute.math.rsqrt(x, fastmath=True)


def fast_rcp(x: Float32) -> Float32:
    """Fast reciprocal.

    Args:
        x: Input value

    Returns:
        1 / x using fast approximation
    """
    if not CUTLASS_AVAILABLE:
        raise RuntimeError("CUTLASS not available")

    # Use rsqrt(x*x) * x for fast reciprocal, or direct division
    return Float32(1.0) / x
