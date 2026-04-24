import pytest
import torch.nn as nn
from pydantic import ValidationError

import dlecosys.models  # side-effect: registers "mlp"
from dlecosys.shared.models import ModelConfig, ModelFactory, register
from dlecosys.models.mlp import MLPConfig


class TestModelFactoryBuild:
    def test_build_returns_nn_module(self):
        model = ModelFactory.build("mlp", {"input_dim": 4})
        assert isinstance(model, nn.Module)

    def test_build_unknown_name_raises(self):
        with pytest.raises(KeyError, match="registry"):
            ModelFactory.build("nonexistent_model_xyz", {})

    def test_build_invalid_params_raises(self):
        with pytest.raises(ValidationError):
            ModelFactory.build("mlp", {"input_dim": -1})

    def test_build_missing_required_param_raises(self):
        with pytest.raises(ValidationError):
            ModelFactory.build("mlp", {})  # input_dim is required

    def test_build_with_full_params(self):
        model = ModelFactory.build(
            "mlp",
            {
                "input_dim": 8,
                "hidden_dims": [32, 16],
                "output_dim": 2,
                "activation": "gelu",
                "dropout": 0.1,
            },
        )
        assert isinstance(model, nn.Module)


class TestModelFactoryAvailable:
    def test_available_includes_mlp(self):
        assert "mlp" in ModelFactory.available()

    def test_available_returns_sorted_list(self):
        names = ModelFactory.available()
        assert names == sorted(names)


class TestRegisterDecorator:
    def test_register_duplicate_raises(self):
        @register("_test_sentinel_a")
        class _ModelA(nn.Module):
            config_class = MLPConfig

            @classmethod
            def from_config(cls, config):
                return cls()

            def forward(self, x):
                return x

        with pytest.raises(ValueError, match="already registered"):
            @register("_test_sentinel_a")
            class _ModelB(nn.Module):
                config_class = MLPConfig

                @classmethod
                def from_config(cls, config):
                    return cls()

                def forward(self, x):
                    return x

    def test_register_missing_config_class_raises(self):
        with pytest.raises(TypeError, match="config_class"):
            @register("_test_sentinel_b")
            class _NoCfg(nn.Module):
                @classmethod
                def from_config(cls, config):
                    return cls()

                def forward(self, x):
                    return x

    def test_register_missing_from_config_raises(self):
        with pytest.raises(TypeError, match="from_config"):
            @register("_test_sentinel_c")
            class _NoFactory(nn.Module):
                config_class = MLPConfig

                def forward(self, x):
                    return x
