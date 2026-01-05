"""RMSNorm kernel tests."""

import pytest
import torch

from mhc.cute_dsl.rmsnorm import rmsnorm, rmsnorm_fwd
from .conftest import get_tolerance, rmsnorm_ref


class TestRMSNormForward:
    """Forward pass correctness tests."""

    @pytest.mark.parametrize("eps", [1e-5, 1e-6])
    @pytest.mark.parametrize(
        "input_dtype", [torch.bfloat16, torch.float16, torch.float32]
    )
    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024, 2048, 4096])
    @pytest.mark.parametrize("M", [1, 32, 128])
    def test_forward_correctness(self, M, N, input_dtype, eps, device, seed):
        """Test forward pass matches reference implementation."""
        atol, rtol = get_tolerance(input_dtype)

        x = torch.randn(M, N, device=device, dtype=input_dtype)
        weight = torch.randn(N, device=device, dtype=torch.float32)

        out = rmsnorm(x, weight, eps=eps)
        out_ref = rmsnorm_ref(x, weight, eps=eps)

        assert out.shape == x.shape
        assert out.dtype == input_dtype
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("N", [192, 760, 1128, 3000])
    def test_uneven_dimensions(self, N, device, seed):
        """Test with non-power-of-2 dimensions."""
        M = 32
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        weight = torch.randn(N, device=device, dtype=torch.float32)

        out = rmsnorm(x, weight)
        out_ref = rmsnorm_ref(x, weight)

        torch.testing.assert_close(out, out_ref, atol=1e-1, rtol=1e-2)

    def test_without_weight(self, device, seed):
        """Test RMSNorm without weight parameter."""
        M, N = 32, 256
        x = torch.randn(M, N, device=device, dtype=torch.float32)

        out = rmsnorm(x, None)
        out_ref = rmsnorm_ref(x, None)

        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


class TestRMSNormBackward:
    """Backward pass correctness tests."""

    @pytest.mark.parametrize("N", [256, 1024, 4096])
    @pytest.mark.parametrize("M", [1, 32, 128])
    def test_backward_correctness(self, M, N, device, seed):
        """Test backward pass produces valid gradients."""
        x = torch.randn(M, N, device=device, dtype=torch.float32, requires_grad=True)
        weight = torch.randn(N, device=device, dtype=torch.float32, requires_grad=True)

        out = rmsnorm(x, weight)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert x.grad is not None
        assert weight.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(weight.grad).any()


class TestRMSNormNumericalStability:
    """Numerical stability tests."""

    def test_large_values(self, device, seed):
        """Test with large input values."""
        M, N = 32, 1024
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16) * 1000
        weight = torch.randn(N, device=device, dtype=torch.float32)

        out = rmsnorm(x, weight)

        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_small_values(self, device, seed):
        """Test with small input values."""
        M, N = 32, 1024
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16) * 1e-6
        weight = torch.randn(N, device=device, dtype=torch.float32)

        out = rmsnorm(x, weight)

        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_mixed_extreme_values(self, device, seed):
        """Test with mixed large and small values."""
        M, N = 32, 1024
        x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        x[:, : N // 2] *= 1000
        x[:, N // 2 :] *= 1e-6
        weight = torch.randn(N, device=device, dtype=torch.float32)

        out = rmsnorm(x, weight)

        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"
