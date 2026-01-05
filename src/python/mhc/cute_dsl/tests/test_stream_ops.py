"""Stream operations kernel tests."""

import pytest
import torch

from mhc.cute_dsl.stream_ops import (
    stream_aggregate,
    stream_distribute_mix_add,
    StreamAggregateFunction,
    StreamDistributeMixAddFunction,
)
from .conftest import stream_aggregate_ref, stream_distribute_mix_add_ref


class TestStreamAggregate:
    """Stream aggregation tests."""

    @pytest.mark.parametrize("B", [1, 8, 32])
    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    @pytest.mark.parametrize("C", [64, 128, 256, 1024])
    def test_forward_shape(self, B, n, C, device, seed):
        """Test output shape is correct."""
        inp = torch.randn(B, n, C, device=device, dtype=torch.float32)
        H_pre = torch.randn(n, device=device, dtype=torch.float32)

        out, H_activated = stream_aggregate(inp, H_pre)

        assert out.shape == (B, C)
        assert H_activated.shape == (n,)

    @pytest.mark.parametrize("n", [2, 4, 8])
    def test_sigmoid_activation(self, n, device, seed):
        """Test H weights are properly sigmoid-activated."""
        B, C = 8, 128
        inp = torch.randn(B, n, C, device=device, dtype=torch.float32)
        H_pre = torch.randn(n, device=device, dtype=torch.float32)

        _, H_activated = stream_aggregate(inp, H_pre)

        # Activated weights should be in (0, 1) due to sigmoid
        assert (H_activated > 0).all()
        assert (H_activated < 1).all()

        # Check against reference sigmoid
        H_ref = torch.sigmoid(H_pre)
        torch.testing.assert_close(H_activated, H_ref, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("B,n,C", [(8, 4, 128), (32, 8, 256)])
    def test_matches_reference(self, B, n, C, device, seed):
        """Test output matches reference implementation."""
        inp = torch.randn(B, n, C, device=device, dtype=torch.float32)
        H_pre = torch.randn(n, device=device, dtype=torch.float32)

        out, _ = stream_aggregate(inp, H_pre)
        out_ref, _ = stream_aggregate_ref(inp, H_pre)

        torch.testing.assert_close(out, out_ref, atol=1e-5, rtol=1e-5)


class TestStreamDistributeMixAdd:
    """Stream distribute/mix/add tests."""

    @pytest.mark.parametrize("B", [1, 8, 32])
    @pytest.mark.parametrize("n", [2, 4, 8])
    @pytest.mark.parametrize("C", [64, 128, 256])
    def test_forward_shape(self, B, n, C, device, seed):
        """Test output shape is correct."""
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)
        y_norm = torch.randn(B, C, device=device, dtype=torch.float32)
        H_post = torch.randn(n, device=device, dtype=torch.float32)
        M = torch.randn(n, n, device=device, dtype=torch.float32)

        out, H_activated = stream_distribute_mix_add(x, y_norm, H_post, M)

        assert out.shape == (B, n, C)
        assert H_activated.shape == (n,)

    @pytest.mark.parametrize("B,n,C", [(8, 4, 128), (32, 8, 256)])
    def test_matches_reference(self, B, n, C, device, seed):
        """Test output matches reference implementation."""
        x = torch.randn(B, n, C, device=device, dtype=torch.float32)
        y_norm = torch.randn(B, C, device=device, dtype=torch.float32)
        H_post = torch.randn(n, device=device, dtype=torch.float32)
        M = torch.randn(n, n, device=device, dtype=torch.float32)

        out, _ = stream_distribute_mix_add(x, y_norm, H_post, M)
        out_ref, _ = stream_distribute_mix_add_ref(x, y_norm, H_post, M)

        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)


class TestStreamOpsBackward:
    """Backward pass tests for stream operations."""

    def test_aggregate_backward(self, device, seed):
        """Test backward pass for stream_aggregate."""
        B, n, C = 8, 4, 128
        inp = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)
        H_pre = torch.randn(n, device=device, dtype=torch.float32, requires_grad=True)

        out, _ = StreamAggregateFunction.apply(inp, H_pre)
        loss = out.sum()
        loss.backward()

        assert inp.grad is not None
        assert H_pre.grad is not None
        assert not torch.isnan(inp.grad).any()
        assert not torch.isnan(H_pre.grad).any()

    def test_distribute_mix_add_backward(self, device, seed):
        """Test backward pass for stream_distribute_mix_add."""
        B, n, C = 8, 4, 128
        x = torch.randn(B, n, C, device=device, dtype=torch.float32, requires_grad=True)
        y_norm = torch.randn(B, C, device=device, dtype=torch.float32, requires_grad=True)
        H_post = torch.randn(n, device=device, dtype=torch.float32, requires_grad=True)
        M = torch.randn(n, n, device=device, dtype=torch.float32, requires_grad=True)

        out, _ = StreamDistributeMixAddFunction.apply(x, y_norm, H_post, M)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert y_norm.grad is not None
        assert H_post.grad is not None
        assert M.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(y_norm.grad).any()
