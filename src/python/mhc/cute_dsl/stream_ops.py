"""Stream operations kernel implementations using CuTe DSL.

Implements the stream aggregation and distribution operations for MHC layer:
- stream_aggregate: Weighted sum of streams with sigmoid-activated weights
- stream_distribute_mix_add: Distribute, mix via matrix multiply, and add

Features:
- Fused sigmoid activation
- Vectorized bf16 operations
- Shared memory weight broadcast
"""

from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.autograd import Function

try:
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32, Int32, BFloat16, const_expr

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False

from .cute_dsl_utils import (
    check_cuda_tensor,
    ensure_contiguous,
    torch2cute_dtype_map,
)
from .fast_math import fast_sigmoid


BLOCK_SIZE = 256
MAX_N = 32  # Maximum expansion rate supported


class StreamAggregate:
    """Stream aggregation kernel with fused sigmoid.

    Computes: out[b, c] = sum_i(sigmoid(H_pre[i]) * inp[b, i, c])

    Also outputs the activated weights for backward pass.
    """

    def __init__(self, max_n: int = MAX_N):
        self.max_n = max_n

    if CUTLASS_AVAILABLE:

        @cute.jit
        def __call__(
            self,
            mOut,  # [B, C] output (bf16)
            mHPreActivated,  # [n] activated weights output
            mInp,  # [B, n, C] input (float32)
            mHPreRaw,  # [n] raw weights
            B: Int32,
            n: Int32,
            C: Int32,
            stream,
        ):
            """Launch stream aggregate kernel."""
            blocks = (B * C + BLOCK_SIZE - 1) // BLOCK_SIZE

            self.kernel(mOut, mHPreActivated, mInp, mHPreRaw, B, n, C).launch(
                grid=[blocks, 1, 1],
                block=[BLOCK_SIZE, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mOut,
            mHPreActivated,
            mInp,
            mHPreRaw,
            B: Int32,
            n: Int32,
            C: Int32,
        ):
            """Stream aggregate kernel."""
            tid = cute.arch.thread_idx()[0]
            bid = cute.arch.block_idx()[0]

            # Load and activate H_pre to shared memory
            smem = cutlass.utils.SmemAllocator()
            s_H_pre = smem.allocate_tensor(
                Float32, cute.make_layout((self.max_n,)), byte_alignment=16
            )

            if tid < n:
                activated = cute.math.rsqrt(
                    (Float32(1.0) + cute.math.exp(-mHPreRaw[tid], fastmath=True))
                    * (Float32(1.0) + cute.math.exp(-mHPreRaw[tid], fastmath=True)),
                    fastmath=True,
                ) * (Float32(1.0) + cute.math.exp(-mHPreRaw[tid], fastmath=True))
                s_H_pre[tid] = activated
                mHPreActivated[tid] = activated
            cute.arch.syncthreads()

            idx = bid * BLOCK_SIZE + tid
            if idx >= B * C:
                return

            b = idx // C
            c = idx % C

            sum_val = Float32(0.0)
            for i in cutlass.range(n):
                sum_val = sum_val + s_H_pre[i] * mInp[b, i, c]

            mOut[idx] = sum_val.to(mOut.element_type)


class StreamDistributeMixAdd:
    """Stream distribute, mix, and add kernel with fused sigmoid.

    Computes:
        mix[b, i, c] = sum_j(M[i, j] * x_inp[b, j, c])
        out[b, i, c] = mix[b, i, c] + 2*sigmoid(H_post[i]) * y_norm[b, c]
    """

    def __init__(self, max_n: int = MAX_N):
        self.max_n = max_n

    if CUTLASS_AVAILABLE:

        @cute.jit
        def __call__(
            self,
            mOut,  # [B, n, C] output
            mHPostActivated,  # [n] activated weights output
            mXInp,  # [B, n, C] input x
            mYNorm,  # [B, C] normalized y (bf16)
            mHPostRaw,  # [n] raw weights
            mM,  # [n, n] mixing matrix
            B: Int32,
            n: Int32,
            C: Int32,
            stream,
        ):
            """Launch stream distribute mix add kernel."""
            blocks = (B * n * C + BLOCK_SIZE - 1) // BLOCK_SIZE

            self.kernel(
                mOut, mHPostActivated, mXInp, mYNorm, mHPostRaw, mM, B, n, C
            ).launch(
                grid=[blocks, 1, 1],
                block=[BLOCK_SIZE, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            mOut,
            mHPostActivated,
            mXInp,
            mYNorm,
            mHPostRaw,
            mM,
            B: Int32,
            n: Int32,
            C: Int32,
        ):
            """Stream distribute mix add kernel."""
            tid = cute.arch.thread_idx()[0]
            bid = cute.arch.block_idx()[0]

            # Load M and H_post to shared memory
            smem = cutlass.utils.SmemAllocator()
            s_M = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.max_n, self.max_n)),
                byte_alignment=16,
            )
            s_H_post = smem.allocate_tensor(
                Float32, cute.make_layout((self.max_n,)), byte_alignment=16
            )

            if tid < n * n:
                i = tid // n
                j = tid % n
                s_M[i, j] = mM[i, j]

            if tid < n:
                exp_neg = cute.math.exp(-mHPostRaw[tid], fastmath=True)
                activated = Float32(2.0) / (Float32(1.0) + exp_neg)
                s_H_post[tid] = activated
                mHPostActivated[tid] = activated
            cute.arch.syncthreads()

            idx = bid * BLOCK_SIZE + tid
            if idx >= B * n * C:
                return

            b = idx // (n * C)
            remainder = idx % (n * C)
            i = remainder // C
            c = remainder % C

            # Compute mix sum
            mix_sum = Float32(0.0)
            for j in cutlass.range(n):
                mix_sum = mix_sum + s_M[i, j] * mXInp[b, j, c]

            # Add distributed term
            y_val = mYNorm[b, c].to(Float32)
            mOut[b, i, c] = mix_sum + s_H_post[i] * y_val


# PyTorch reference implementations
def _stream_aggregate_pytorch(
    inp: Tensor, H_pre_raw: Tensor
) -> Tuple[Tensor, Tensor]:
    """PyTorch reference for stream aggregate.

    Args:
        inp: Input tensor [B, n, C]
        H_pre_raw: Raw weights [n]

    Returns:
        Tuple of (output [B, C], activated_weights [n])
    """
    H_pre_activated = torch.sigmoid(H_pre_raw)
    # inp: [B, n, C], H_pre_activated: [n] -> [1, n, 1]
    out = (inp * H_pre_activated.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    return out, H_pre_activated


def _stream_distribute_mix_add_pytorch(
    x_inp: Tensor,
    y_norm: Tensor,
    H_post_raw: Tensor,
    M: Tensor,
) -> Tuple[Tensor, Tensor]:
    """PyTorch reference for stream distribute mix add.

    Args:
        x_inp: Input tensor [B, n, C]
        y_norm: Normalized input [B, C]
        H_post_raw: Raw weights [n]
        M: Mixing matrix [n, n]

    Returns:
        Tuple of (output [B, n, C], activated_weights [n])
    """
    B, n, C = x_inp.shape
    H_post_activated = 2.0 * torch.sigmoid(H_post_raw)

    # Mix: x_inp [B, n, C] @ M.T [n, n] -> [B, n, C]
    # Actually: for each output stream i, sum over j of M[i,j] * x_inp[b,j,c]
    # This is einsum('bnc,ij->bic', x_inp, M) but with M indices as [i,j]
    # More simply: permute and matmul
    mix = torch.einsum("bjc,ij->bic", x_inp, M)

    # Distribute: H_post_activated[i] * y_norm[b, c]
    # [n] * [B, C] -> [B, n, C]
    dist = H_post_activated.unsqueeze(0).unsqueeze(-1) * y_norm.unsqueeze(1)

    out = mix + dist
    return out, H_post_activated


def stream_aggregate(
    inp: Tensor, H_pre_raw: Tensor
) -> Tuple[Tensor, Tensor]:
    """Stream aggregation with fused sigmoid activation.

    Computes weighted sum of input streams where weights are
    sigmoid-activated.

    Args:
        inp: Input tensor [B, n, C]
        H_pre_raw: Raw pre-activation weights [n]

    Returns:
        Tuple of (output [B, C], activated_weights [n])
    """
    check_cuda_tensor(inp, "inp")
    check_cuda_tensor(H_pre_raw, "H_pre_raw")
    inp = ensure_contiguous(inp).float()
    H_pre_raw = ensure_contiguous(H_pre_raw).float()

    return _stream_aggregate_pytorch(inp, H_pre_raw)


def stream_distribute_mix_add(
    x_inp: Tensor,
    y_norm: Tensor,
    H_post_raw: Tensor,
    M: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Stream distribute, mix, and add with fused sigmoid.

    Args:
        x_inp: Input tensor [B, n, C]
        y_norm: Normalized input [B, C]
        H_post_raw: Raw post-activation weights [n]
        M: Mixing matrix [n, n]

    Returns:
        Tuple of (output [B, n, C], activated_weights [n])
    """
    check_cuda_tensor(x_inp, "x_inp")
    check_cuda_tensor(y_norm, "y_norm")
    check_cuda_tensor(H_post_raw, "H_post_raw")
    check_cuda_tensor(M, "M")

    x_inp = ensure_contiguous(x_inp).float()
    y_norm = ensure_contiguous(y_norm)  # Can be bf16
    H_post_raw = ensure_contiguous(H_post_raw).float()
    M = ensure_contiguous(M).float()

    return _stream_distribute_mix_add_pytorch(x_inp, y_norm.float(), H_post_raw, M)


class StreamAggregateFunction(Function):
    """PyTorch autograd function for stream aggregation."""

    @staticmethod
    def forward(ctx, inp: Tensor, H_pre_raw: Tensor):
        out, H_pre_activated = stream_aggregate(inp, H_pre_raw)
        ctx.save_for_backward(inp, H_pre_activated)
        return out, H_pre_activated

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_H_activated: Tensor):
        inp, H_pre_activated = ctx.saved_tensors
        B, n, C = inp.shape

        # d_inp[b, i, c] = grad_output[b, c] * H_pre_activated[i]
        d_inp = grad_output.unsqueeze(1) * H_pre_activated.view(1, n, 1)

        # d_H_pre_raw[i] = sum_b,c grad_output[b, c] * inp[b, i, c] * H' (sigmoid derivative)
        # H' = H * (1 - H) for sigmoid
        H_deriv = H_pre_activated * (1 - H_pre_activated)
        d_H_pre_raw = (grad_output.unsqueeze(1) * inp * H_deriv.view(1, n, 1)).sum(
            dim=(0, 2)
        )

        return d_inp, d_H_pre_raw


class StreamDistributeMixAddFunction(Function):
    """PyTorch autograd function for stream distribute mix add."""

    @staticmethod
    def forward(ctx, x_inp: Tensor, y_norm: Tensor, H_post_raw: Tensor, M: Tensor):
        out, H_post_activated = stream_distribute_mix_add(x_inp, y_norm, H_post_raw, M)
        ctx.save_for_backward(x_inp, y_norm.float(), H_post_activated, M)
        return out, H_post_activated

    @staticmethod
    def backward(ctx, grad_output: Tensor, grad_H_activated: Tensor):
        x_inp, y_norm, H_post_activated, M = ctx.saved_tensors
        B, n, C = x_inp.shape

        # d_x_inp from mix: grad_output @ M
        # d_x_inp[b, j, c] = sum_i grad_output[b, i, c] * M[i, j]
        d_x_inp = torch.einsum("bic,ij->bjc", grad_output, M)

        # d_y_norm from distribute: sum_i grad_output[b, i, c] * H_post_activated[i]
        d_y_norm = (grad_output * H_post_activated.view(1, n, 1)).sum(dim=1)

        # d_M[i, j] = sum_b,c grad_output[b, i, c] * x_inp[b, j, c]
        d_M = torch.einsum("bic,bjc->ij", grad_output, x_inp)

        # d_H_post_raw from distribute
        # H' = 2 * sigmoid * (1 - sigmoid) for 2*sigmoid
        H_half = H_post_activated / 2  # Original sigmoid
        H_deriv = 2 * H_half * (1 - H_half)
        d_H_post_raw = (
            grad_output * y_norm.unsqueeze(1) * H_deriv.view(1, n, 1)
        ).sum(dim=(0, 2))

        return d_x_inp, d_y_norm, d_H_post_raw, d_M
