"""Full MHC layer integration tests."""

import pytest
import torch

from mhc.cute_dsl.mhc_layer import MHCLayerDSL, mhc_layer_fused_dsl


class TestMHCLayerForward:
    """Forward pass integration tests."""

    @pytest.mark.parametrize("B", [1, 8, 32])
    @pytest.mark.parametrize("n", [2, 4, 8])
    @pytest.mark.parametrize("C", [64, 128, 256])
    def test_forward_shape(self, B, n, C, device, seed):
        """Test full layer forward produces correct shape."""
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device)

        out = layer(x)

        assert out.shape == (B, n, C)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_deterministic(self, device, seed):
        """Test layer is deterministic."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device)

        out1 = layer(x)
        out2 = layer(x)

        torch.testing.assert_close(out1, out2)


class TestMHCLayerBackward:
    """Backward pass integration tests."""

    def test_gradient_flow(self, device, seed):
        """Test gradients flow through all parameters."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.norm() > 0, "Input gradient should be non-zero"

        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"


class TestMHCLayerFunctional:
    """Tests for functional interface."""

    @pytest.mark.parametrize("B,n,C", [(8, 4, 128), (32, 8, 256)])
    def test_functional_matches_module(self, B, n, C, device, seed):
        """Test functional interface matches module interface."""
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device)

        out_module = layer(x)
        out_functional = mhc_layer_fused_dsl(
            x,
            layer.rmsnorm_weight,
            layer.H_pre,
            layer.H_post,
            layer.H_res,
            layer.sinkhorn_iters,
            layer.eps,
        )

        torch.testing.assert_close(out_module, out_functional)


class TestMHCLayerNumericalStability:
    """Numerical stability tests."""

    def test_large_input(self, device, seed):
        """Test with large input values."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device) * 100

        out = layer(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_small_input(self, device, seed):
        """Test with small input values."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device) * 1e-6

        out = layer(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_many_sinkhorn_iterations(self, device, seed):
        """Test with many Sinkhorn iterations."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(
            hidden_dim=C, expansion_rate=n, sinkhorn_iters=100
        ).to(device)
        x = torch.randn(B, n, C, device=device)

        out = layer(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestMHCLayerSerialization:
    """Test model serialization."""

    def test_save_load(self, device, seed, tmp_path):
        """Test saving and loading layer."""
        B, n, C = 8, 4, 128
        layer = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        x = torch.randn(B, n, C, device=device)

        # Get original output
        out_original = layer(x)

        # Save and load
        path = tmp_path / "layer.pt"
        torch.save(layer.state_dict(), path)

        layer_loaded = MHCLayerDSL(hidden_dim=C, expansion_rate=n).to(device)
        layer_loaded.load_state_dict(torch.load(path))

        out_loaded = layer_loaded(x)

        torch.testing.assert_close(out_original, out_loaded)
