"""
MLP — Multilayer Perceptron
----------------------------
Reference implementation of the dlecosys model convention.
Registered as "mlp" in the global model registry.

Classes
-------
MLPConfig
    Pydantic config: input/output dims, hidden widths, activation, dropout.
MLP
    Feed-forward MLP that follows the ModelConfig + from_config convention.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from pydantic import Field, field_validator

from dlecosys.shared.models import ModelConfig, register

__all__ = ["MLP", "MLPConfig"]

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


class MLPConfig(ModelConfig):
    """
    Configuration for the MLP reference model.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int, default [128, 64]
        Width of each hidden layer, in order.
    output_dim : int, default 1
        Number of output units.
    activation : str, default "relu"
        Activation applied after each hidden layer.
        One of: relu, gelu, tanh, silu, leaky_relu.
    dropout : float, default 0.0
        Dropout probability applied after each activation (0.0 = disabled).
    """

    input_dim: int
    hidden_dims: List[int] = Field(default_factory=lambda: [128, 64])
    output_dim: int = 1
    activation: str = "relu"
    dropout: float = 0.0

    @field_validator("input_dim", "output_dim")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"dimension must be >= 1; got {v}")
        return v

    @field_validator("hidden_dims")
    @classmethod
    def _positive_hidden(cls, v: List[int]) -> List[int]:
        for d in v:
            if d < 1:
                raise ValueError(f"every hidden_dim must be >= 1; got {d}")
        return v

    @field_validator("activation")
    @classmethod
    def _valid_activation(cls, v: str) -> str:
        if v not in _ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(_ACTIVATIONS)}; got {v!r}"
            )
        return v

    @field_validator("dropout")
    @classmethod
    def _valid_dropout(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {v}")
        return v


@register("mlp")
class MLP(nn.Module):
    """
    Feed-forward multilayer perceptron.

    Demo network bundled with the repo for pipeline validation and synthetic
    experiments. Not intended as a template — the required interface
    (``config_class``, ``from_config``, ``@register``) is enforced by the
    registry decorator for every model.
    """

    config_class = MLPConfig

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        act_cls = _ACTIVATIONS[config.activation]
        dims = [config.input_dim] + list(config.hidden_dims)
        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(act_cls())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))

        layers.append(nn.Linear(dims[-1], config.output_dim))
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_config(cls, config: MLPConfig) -> "MLP":
        return cls(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
