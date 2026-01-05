"""Complete MHC Layer implementation using CuTe DSL kernels.

This module provides the full MHC (Multi-Head Communication) layer
using CuTe DSL kernels for all custom operations:
- RMSNorm normalization
- Sinkhorn-Knopp doubly-stochastic normalization
- Stream aggregation and distribution

The layer implements the mHC algorithm from:
"mHC: Manifold-Constrained Hyper-Connections" (DeepSeek-AI, 2025)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from .rmsnorm import rmsnorm, rmsnorm_fwd, RMSNormFunction
from .sinkhorn import sinkhorn_knopp, sinkhorn_knopp_fwd, SinkhornKnoppFunction
from .stream_ops import (
    stream_aggregate,
    stream_distribute_mix_add,
    StreamAggregateFunction,
    StreamDistributeMixAddFunction,
)
from .cute_dsl_utils import check_cuda_tensor, ensure_contiguous


class MHCLayerFunction(Function):
    """Fused MHC layer forward/backward using CuTe DSL kernels."""

    @staticmethod
    def forward(
        ctx,
        x_expanded: Tensor,
        rmsnorm_weight: Tensor,
        H_pre: Tensor,
        H_post: Tensor,
        H_res: Tensor,
        sinkhorn_iters: int,
        eps: float,
    ) -> Tensor:
        """Forward pass of MHC layer.

        Args:
            x_expanded: Input tensor [B, n, C]
            rmsnorm_weight: RMSNorm weights [C]
            H_pre: Pre-aggregation weights [n]
            H_post: Post-distribution weights [n]
            H_res: Residual mixing weights [n, n]
            sinkhorn_iters: Number of Sinkhorn-Knopp iterations
            eps: Epsilon for numerical stability

        Returns:
            Output tensor [B, n, C]
        """
        B, n, C = x_expanded.shape

        # Step 1: Stream aggregation with sigmoid activation
        # x_agg[b, c] = sum_i sigmoid(H_pre[i]) * x_expanded[b, i, c]
        x_agg, H_pre_activated = stream_aggregate(x_expanded, H_pre)

        # Step 2: RMSNorm on aggregated features
        # y_norm = rmsnorm(x_agg)
        y_norm, rms = rmsnorm_fwd(
            x_agg,  # [B, C]
            rmsnorm_weight,
            out_dtype=torch.bfloat16,
            eps=eps,
            store_rstd=True,
        )
        # y_norm shape: [B, C]

        # Step 3: Sinkhorn-Knopp on residual weights
        # M = sinkhorn(exp(H_res))
        H_res_exp = torch.exp(H_res)
        M = sinkhorn_knopp_fwd(H_res_exp, num_iters=sinkhorn_iters, eps=eps)

        # Step 4: Stream distribute, mix, and add
        # out[b, i, c] = sum_j M[i,j] * x_expanded[b,j,c] + 2*sigmoid(H_post[i]) * y_norm[b,c]
        output, H_post_activated = stream_distribute_mix_add(
            x_expanded.float(), y_norm, H_post, M
        )

        # Save for backward
        ctx.save_for_backward(
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg.bfloat16(),
            H_pre_activated,
            H_post_activated,
            M,
            y_norm.bfloat16(),
            H_res,
        )
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Backward pass of MHC layer."""
        (
            x_expanded,
            rmsnorm_weight,
            rms,
            x_agg_bf16,
            H_pre_activated,
            H_post_activated,
            M,
            y_norm_bf16,
            H_res,
        ) = ctx.saved_tensors

        B, n, C = x_expanded.shape
        grad_output = grad_output.float()

        # Backward through stream distribute mix add
        # d_x_from_mix, d_y_norm, d_M, d_H_post_raw
        d_x_mix = torch.einsum("bic,ij->bjc", grad_output, M)

        H_half = H_post_activated / 2
        H_post_deriv = 2 * H_half * (1 - H_half)
        d_y_norm = (grad_output * H_post_activated.view(1, n, 1)).sum(dim=1)

        d_M = torch.einsum("bic,bjc->ij", grad_output, x_expanded.float())
        d_H_post = (
            grad_output * y_norm_bf16.float().unsqueeze(1) * H_post_deriv.view(1, n, 1)
        ).sum(dim=(0, 2))

        # Backward through Sinkhorn (approximate)
        # d_H_res ≈ d_M * M (simplified)
        H_res_exp = torch.exp(H_res)
        d_H_res = d_M * H_res_exp  # Gradient through exp

        # Backward through RMSNorm
        y_norm_f32 = y_norm_bf16.float()
        x_agg_f32 = x_agg_bf16.float()
        rstd = (1.0 / rms).squeeze()

        # Expand d_y_norm back
        x_norm = x_agg_f32 * rstd.unsqueeze(1)
        d_x_norm = d_y_norm * rmsnorm_weight.float()

        # RMSNorm backward
        dot = (d_x_norm * x_norm).sum(dim=-1, keepdim=True) / C
        d_x_agg = (d_x_norm - x_norm * dot) * rstd.unsqueeze(1)
        d_rmsnorm_weight = (d_y_norm.unsqueeze(0) * x_norm.unsqueeze(0)).sum(dim=(0, 1))

        # Backward through stream aggregate
        d_x_agg_expanded = d_x_agg.unsqueeze(1) * H_pre_activated.view(1, n, 1)

        H_pre_deriv = H_pre_activated * (1 - H_pre_activated)
        d_H_pre = (
            d_x_agg.unsqueeze(1) * x_expanded.float() * H_pre_deriv.view(1, n, 1)
        ).sum(dim=(0, 2))

        # Combine gradients
        d_x = d_x_mix + d_x_agg_expanded

        return d_x, d_rmsnorm_weight, d_H_pre, d_H_post, d_H_res, None, None


def mhc_layer_fused_dsl(
    x_expanded: Tensor,
    rmsnorm_weight: Tensor,
    H_pre: Tensor,
    H_post: Tensor,
    H_res: Tensor,
    sinkhorn_iters: int = 20,
    eps: float = 1e-5,
) -> Tensor:
    """Functional interface for MHC layer using CuTe DSL.

    Args:
        x_expanded: Input tensor [B, n, C]
        rmsnorm_weight: RMSNorm weights [C]
        H_pre: Pre-aggregation weights [n]
        H_post: Post-distribution weights [n]
        H_res: Residual mixing weights [n, n]
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations
        eps: Epsilon for numerical stability

    Returns:
        Output tensor [B, n, C]
    """
    return MHCLayerFunction.apply(
        x_expanded.float(),
        rmsnorm_weight.bfloat16(),
        H_pre.float(),
        H_post.float(),
        H_res.float(),
        sinkhorn_iters,
        eps,
    )


class MHCLayerDSL(nn.Module):
    """MHC Layer using CuTe DSL kernels.

    This is a drop-in replacement for the CUDA-based MHCLayer,
    implemented entirely using CuTe DSL for all custom operations.

    Args:
        hidden_dim: The hidden dimension (C).
        expansion_rate: The expansion rate (n).
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations.
        eps: Epsilon for numerical stability.
        alpha_init: Initialization scale for alpha parameters.
        use_dynamic_h: If True, uses per-batch H values (not yet supported).
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        alpha_init: float = 0.01,
        use_dynamic_h: bool = False,
    ):
        super().__init__()

        if use_dynamic_h:
            raise NotImplementedError(
                "Dynamic H not yet implemented in CuTe DSL version. "
                "Use use_dynamic_h=False or the CUDA version."
            )

        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        n = expansion_rate
        C = hidden_dim

        # RMSNorm weight
        self.rmsnorm_weight = nn.Parameter(torch.ones(hidden_dim, dtype=torch.bfloat16))

        # Static H parameters
        self.H_pre = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        self.H_post = nn.Parameter(torch.zeros(n, dtype=torch.float32))
        H_res_init = alpha_init * torch.randn(n, n)
        self.H_res = nn.Parameter(H_res_init.float())

    def forward(self, x_expanded: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_expanded: Input tensor [B, n, C]

        Returns:
            Output tensor [B, n, C]
        """
        B, n, C = x_expanded.shape
        assert n == self.expansion_rate, f"Expected n={self.expansion_rate}, got {n}"
        assert C == self.hidden_dim, f"Expected C={self.hidden_dim}, got {C}"

        return mhc_layer_fused_dsl(
            x_expanded,
            self.rmsnorm_weight,
            self.H_pre,
            self.H_post,
            self.H_res,
            self.sinkhorn_iters,
            self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"expansion_rate={self.expansion_rate}, "
            f"sinkhorn_iters={self.sinkhorn_iters}"
        )


# Aliases for backwards compatibility
MHCLayerCuTe = MHCLayerDSL
mhc_layer_cute = mhc_layer_fused_dsl
