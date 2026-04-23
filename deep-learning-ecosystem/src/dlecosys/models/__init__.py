"""
dlecosys Model Zoo
-------------------
Bundled model implementations for pipeline validation and synthetic experiments.

Import this module to register bundled models with the global model registry.
User models are registered independently via ``@register`` — they are not
derived from or modelled after anything here.

Classes
-------
MLP
    Feed-forward multilayer perceptron. Demo network.
MLPConfig
    Pydantic configuration for the MLP.
"""

from dlecosys.models.mlp import MLP, MLPConfig

__all__ = ["MLP", "MLPConfig"]
