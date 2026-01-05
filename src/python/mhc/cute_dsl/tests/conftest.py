"""Shared pytest fixtures for CuTe DSL tests."""

import pytest
import torch


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42


def get_tolerance(dtype: torch.dtype) -> tuple:
    """Get appropriate tolerance for dtype.

    Returns:
        Tuple of (atol, rtol)
    """
    if dtype == torch.bfloat16:
        return 1e-1, 1e-2
    elif dtype == torch.float16:
        return 1e-2, 1e-3
    elif dtype == torch.float32:
        return 1e-4, 1e-4
    else:
        return 1e-5, 1e-5


def rmsnorm_ref(x, weight=None, eps=1e-6):
    """Reference implementation for RMSNorm.

    Args:
        x: Input tensor [B, N]
        weight: Optional weight tensor [N]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor
    """
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
    x_norm = x_f32 / rms
    if weight is not None:
        out = x_norm * weight.float()
    else:
        out = x_norm
    return out.to(x.dtype)


def sinkhorn_ref(M, num_iters=10, eps=1e-8):
    """Reference implementation for Sinkhorn-Knopp.

    Args:
        M: Input matrix [M, N] or [B, M, N]
        num_iters: Number of iterations
        eps: Epsilon for numerical stability

    Returns:
        Doubly-stochastic matrix
    """
    M = M.float().clone()
    if M.dim() == 2:
        for _ in range(num_iters):
            M = M / (M.sum(dim=1, keepdim=True) + eps)
            M = M / (M.sum(dim=0, keepdim=True) + eps)
    else:
        for _ in range(num_iters):
            M = M / (M.sum(dim=2, keepdim=True) + eps)
            M = M / (M.sum(dim=1, keepdim=True) + eps)
    return M


def stream_aggregate_ref(inp, H_pre):
    """Reference implementation for stream aggregation.

    Args:
        inp: Input tensor [B, n, C]
        H_pre: Pre-activation weights [n]

    Returns:
        Tuple of (output [B, C], activated_weights [n])
    """
    H_pre_activated = torch.sigmoid(H_pre)
    out = (inp * H_pre_activated.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    return out, H_pre_activated


def stream_distribute_mix_add_ref(x_inp, y_norm, H_post, M):
    """Reference implementation for stream distribute mix add.

    Args:
        x_inp: Input tensor [B, n, C]
        y_norm: Normalized input [B, C]
        H_post: Post-activation weights [n]
        M: Mixing matrix [n, n]

    Returns:
        Tuple of (output [B, n, C], activated_weights [n])
    """
    H_post_activated = 2.0 * torch.sigmoid(H_post)
    mix = torch.einsum("bjc,ij->bic", x_inp, M)
    dist = H_post_activated.unsqueeze(0).unsqueeze(-1) * y_norm.unsqueeze(1)
    return mix + dist, H_post_activated
