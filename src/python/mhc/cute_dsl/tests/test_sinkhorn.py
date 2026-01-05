"""Sinkhorn-Knopp kernel tests."""

import pytest
import torch

from mhc.cute_dsl.sinkhorn import sinkhorn_knopp, sinkhorn_knopp_fwd
from .conftest import sinkhorn_ref


class TestSinkhornForward:
    """Forward pass correctness tests."""

    @pytest.mark.parametrize("size", [8, 16, 32, 48, 64])
    @pytest.mark.parametrize("num_iters", [5, 10, 20, 50])
    def test_doubly_stochastic_property(self, size, num_iters, device, seed):
        """Test output is doubly stochastic matrix."""
        M = torch.rand(size, size, device=device) + 0.1

        out = sinkhorn_knopp(M, num_iters=num_iters)

        row_sums = out.sum(dim=1)
        col_sums = out.sum(dim=0)

        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
        )
        torch.testing.assert_close(
            col_sums, torch.ones_like(col_sums), atol=1e-3, rtol=1e-3
        )
        assert (out >= 0).all(), "Output should be non-negative"

    @pytest.mark.parametrize("size", [16, 32, 64])
    def test_matches_reference(self, size, device, seed):
        """Test output matches reference implementation."""
        M = torch.rand(size, size, device=device) + 0.1

        out = sinkhorn_knopp(M, num_iters=20)
        out_ref = sinkhorn_ref(M, num_iters=20)

        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("M,N", [(16, 32), (32, 16), (24, 48)])
    def test_non_square_matrices(self, M, N, device, seed):
        """Test with non-square matrices."""
        mat = torch.rand(M, N, device=device) + 0.1

        out = sinkhorn_knopp(mat, num_iters=20)

        assert out.shape == (M, N)
        assert not torch.isnan(out).any()


class TestSinkhornBackward:
    """Backward pass tests."""

    @pytest.mark.parametrize("size", [8, 16, 32])
    def test_gradient_exists(self, size, device, seed):
        """Test that gradients flow through."""
        M = (torch.rand(size, size, device=device) + 0.1).requires_grad_(True)

        out = sinkhorn_knopp(M, num_iters=10)
        loss = out.sum()
        loss.backward()

        assert M.grad is not None
        assert not torch.isnan(M.grad).any()
        assert not torch.isinf(M.grad).any()


class TestSinkhornNumericalStability:
    """Numerical stability tests."""

    def test_large_values(self, device, seed):
        """Test stability with large input values."""
        size = 32
        M = torch.rand(size, size, device=device) * 1000

        out = sinkhorn_knopp(M, num_iters=50)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_small_values(self, device, seed):
        """Test stability with small input values."""
        size = 32
        M = torch.rand(size, size, device=device) * 1e-6 + 1e-8

        out = sinkhorn_knopp(M, num_iters=20)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestSinkhornBatched:
    """Batched operation tests."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_batched_forward(self, batch_size, device, seed):
        """Test batched forward pass."""
        size = 32
        M = torch.rand(batch_size, size, size, device=device) + 0.1

        out = sinkhorn_knopp(M, num_iters=20)

        assert out.shape == (batch_size, size, size)

        # Check each batch element is doubly stochastic
        for i in range(batch_size):
            row_sums = out[i].sum(dim=1)
            col_sums = out[i].sum(dim=0)
            torch.testing.assert_close(
                row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
            )
            torch.testing.assert_close(
                col_sums, torch.ones_like(col_sums), atol=1e-3, rtol=1e-3
            )
