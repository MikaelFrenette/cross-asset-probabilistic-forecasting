"""
density_model Model Zoo
-----------------------
Bundled model implementations. Importing this module registers each
bundled model with the global model registry via its ``@register("name")``
decorator.

Classes
-------
BasePanelModel
    Abstract base for panel forecasting models.
CausalCrossAssetTransformer
    Causal transformer with cross-asset attention on the final timestep;
    registered as ``"causal_cross_asset_transformer"``.
CausalCrossAssetTransformerConfig
    Pydantic configuration shared by every panel transformer variant.
CausalNextStepTransformer
    GPT-style per-position causal next-step density forecaster with
    cross-asset attention at every timestep; registered as
    ``"causal_next_step_transformer"``.
TemporalOnlyTransformer
    Causal transformer without cross-asset attention; registered as
    ``"temporal_only_transformer"``.
"""

from density_model.models.causal_next_step import CausalNextStepTransformer
from density_model.models.cross_asset import (
    CausalCrossAssetTransformer,
    CausalCrossAssetTransformerConfig,
)
from density_model.models.panel_base import BasePanelModel
from density_model.models.temporal_only import TemporalOnlyTransformer

__all__ = [
    "BasePanelModel",
    "CausalCrossAssetTransformer",
    "CausalCrossAssetTransformerConfig",
    "CausalNextStepTransformer",
    "TemporalOnlyTransformer",
]
