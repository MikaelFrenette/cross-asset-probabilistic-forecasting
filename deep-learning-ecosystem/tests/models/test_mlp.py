import pytest
import torch
from pydantic import ValidationError

from dlecosys.models.mlp import MLP, MLPConfig


def _model(input_dim=4, **kwargs) -> MLP:
    config = MLPConfig(input_dim=input_dim, **kwargs)
    return MLP.from_config(config)


class TestMLPConfig:
    def test_defaults(self):
        cfg = MLPConfig(input_dim=4)
        assert cfg.hidden_dims == [128, 64]
        assert cfg.output_dim == 1
        assert cfg.activation == "relu"
        assert cfg.dropout == 0.0

    def test_invalid_input_dim_raises(self):
        with pytest.raises(ValidationError):
            MLPConfig(input_dim=0)

    def test_invalid_output_dim_raises(self):
        with pytest.raises(ValidationError):
            MLPConfig(input_dim=4, output_dim=0)

    def test_invalid_hidden_dim_raises(self):
        with pytest.raises(ValidationError):
            MLPConfig(input_dim=4, hidden_dims=[64, -1])

    def test_invalid_activation_raises(self):
        with pytest.raises(ValidationError):
            MLPConfig(input_dim=4, activation="swish")

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValidationError):
            MLPConfig(input_dim=4, dropout=1.0)

    def test_dropout_zero_is_valid(self):
        cfg = MLPConfig(input_dim=4, dropout=0.0)
        assert cfg.dropout == 0.0


class TestMLPForward:
    def test_output_shape_default(self):
        model = _model(input_dim=4)
        x = torch.randn(8, 4)
        assert model(x).shape == (8, 1)

    def test_output_shape_custom(self):
        model = _model(input_dim=10, hidden_dims=[32], output_dim=3)
        x = torch.randn(5, 10)
        assert model(x).shape == (5, 3)

    def test_all_activations_build(self):
        for act in ("relu", "gelu", "tanh", "silu", "leaky_relu"):
            model = _model(input_dim=4, activation=act)
            out = model(torch.randn(2, 4))
            assert out.shape == (2, 1)

    def test_deep_network(self):
        model = _model(input_dim=16, hidden_dims=[64, 64, 32, 16], output_dim=2)
        out = model(torch.randn(4, 16))
        assert out.shape == (4, 2)

    def test_no_dropout_in_eval_equals_train(self):
        torch.manual_seed(0)
        model = _model(input_dim=4, dropout=0.0)
        x = torch.randn(16, 4)
        model.train()
        out_train = model(x)
        model.eval()
        out_eval = model(x)
        assert torch.allclose(out_train, out_eval)

    def test_dropout_active_in_train_mode(self):
        torch.manual_seed(0)
        model = _model(input_dim=32, hidden_dims=[128], dropout=0.5)
        x = torch.randn(64, 32)
        model.train()
        out1 = model(x)
        out2 = model(x)
        # Two stochastic forward passes should differ (with overwhelming probability)
        assert not torch.allclose(out1, out2)

    def test_dropout_deterministic_in_eval_mode(self):
        model = _model(input_dim=4, hidden_dims=[16], dropout=0.5)
        x = torch.randn(8, 4)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))

    def test_from_config_produces_identical_arch(self):
        cfg = MLPConfig(input_dim=4, hidden_dims=[16], output_dim=2)
        m1 = MLP.from_config(cfg)
        m2 = MLP(cfg)
        # Same architecture → same parameter count
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 == p2
