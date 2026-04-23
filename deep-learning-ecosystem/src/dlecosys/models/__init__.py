"""
dlecosys Model Zoo
-------------------
Bundled model implementations. Importing this module registers each bundled
model with the global model registry via its ``@register("name")`` decorator.

Classes
-------
MLP
    Feed-forward multilayer perceptron. Demo network (tabular).
MLPConfig
    Pydantic configuration for the MLP.
BasePanelModel
    Abstract base for panel forecasting models.
CausalCrossAssetTransformer
    Causal transformer with cross-asset attention; registered as
    ``"causal_cross_asset_transformer"``.
CausalCrossAssetTransformerConfig
    Pydantic configuration for the causal cross-asset transformer.
TemporalOnlyTransformer
    Causal transformer without cross-asset attention; registered as
    ``"temporal_only_transformer"``.
"""

from dlecosys.models.cross_asset import (
    CausalCrossAssetTransformer,
    CausalCrossAssetTransformerConfig,
)
from dlecosys.models.mlp import MLP, MLPConfig
from dlecosys.models.panel_base import BasePanelModel
from dlecosys.models.temporal_only import TemporalOnlyTransformer

__all__ = [
    "BasePanelModel",
    "CausalCrossAssetTransformer",
    "CausalCrossAssetTransformerConfig",
    "MLP",
    "MLPConfig",
    "TemporalOnlyTransformer",
]
