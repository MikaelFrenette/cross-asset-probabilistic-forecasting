"""
Model Configuration
-------------------
Pydantic model configurations for panel forecasting architectures.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["CausalCrossAssetTransformerConfig"]


class CausalCrossAssetTransformerConfig(BaseModel):
    """
    Configuration for the causal cross-asset transformer.

    Parameters
    ----------
    d_model : int
        Shared hidden width used after stream fusion.
    dim_feedforward : int
        Feed-forward width used inside each transformer layer.
    num_heads : int
        Number of attention heads used in temporal and cross-asset attention.
    num_layers : int
        Number of causal temporal transformer layers.
    dropout : float
        Dropout rate used across the architecture.
    fusion_mode : {"sum", "concat"}, default="concat"
        Fusion strategy for optional input streams.
    out_steps : int, default=1
        Number of forecast steps predicted by the head.
    target_dim : int, default=1
        Number of target channels predicted per asset.
    dynamic_categorical_cardinalities : tuple of int or None, default=None
        Cardinalities for dynamic categorical features.
    static_categorical_cardinalities : tuple of int or None, default=None
        Cardinalities for static categorical features.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    d_model: int = Field(gt=0)
    dim_feedforward: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    fusion_mode: Literal["sum", "concat"] = "concat"
    out_steps: int = Field(default=1, gt=0)
    target_dim: int = Field(default=1, gt=0)
    dynamic_categorical_cardinalities: tuple[int, ...] | None = None
    static_categorical_cardinalities: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def validate_heads(self) -> CausalCrossAssetTransformerConfig:
        """
        Validate attention-head compatibility with the hidden width.

        Returns
        -------
        CausalCrossAssetTransformerConfig
            Validated model configuration.
        """

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        return self
