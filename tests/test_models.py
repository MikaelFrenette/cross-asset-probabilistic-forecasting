"""
Model Tests
-----------
Unit tests for the causal cross-asset transformer and panel batch schema.
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

torch = pytest.importorskip("torch")

from density_model.config.config_schema import FeatureConfig, ModelConfig
from density_model.models import (
    CausalCrossAssetTransformer,
    CausalCrossAssetTransformerConfig,
    PanelBatch,
    build_causal_cross_asset_transformer,
    build_panel_model,
)
from density_model.preprocessing import PreprocessingBundle

__all__ = []


def build_panel_batch() -> PanelBatch:
    """
    Build a minimal valid panel batch for model tests.

    Returns
    -------
    PanelBatch
        Validated panel batch.
    """

    return PanelBatch(
        X_continuous=torch.randn(2, 3, 4, 2, dtype=torch.float32),
        X_continuous_mask=torch.ones(2, 3, 4, 2, dtype=torch.bool),
        X_cat_continuous=torch.ones(2, 3, 4, 1, dtype=torch.long),
        X_cat_continuous_mask=torch.ones(2, 3, 4, 1, dtype=torch.bool),
        X_cat_static=torch.zeros(2, 3, 4, 1, dtype=torch.long),
        X_cat_static_mask=torch.ones(2, 3, 4, 1, dtype=torch.bool),
        y=torch.zeros(2, 3, 1, 1, dtype=torch.float32),
        y_mask=torch.ones(2, 3, 1, 1, dtype=torch.bool),
        id_index=["A", "B", "C"],
        forecast_start_dates=[np.datetime64("2024-01-03"), np.datetime64("2024-01-04")],
        forecast_end_dates=[np.datetime64("2024-01-03"), np.datetime64("2024-01-04")],
    )


def test_causal_cross_asset_transformer_produces_forecast_shape() -> None:
    """
    Produce predictions shaped like the target tensor.

    Returns
    -------
    None
        This test asserts the model output contract.
    """

    batch = build_panel_batch()
    config = CausalCrossAssetTransformerConfig(
        d_model=8,
        dim_feedforward=16,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
        fusion_mode="concat",
        out_steps=1,
        target_dim=1,
        dynamic_categorical_cardinalities=(4,),
        static_categorical_cardinalities=(3,),
    )
    model = CausalCrossAssetTransformer(
        config,
        continuous_dim=2,
        dynamic_categorical_dim=1,
        static_categorical_dim=1,
    )

    output = model(batch)
    assert output.shape == torch.Size([2, 3, 1, 2])
    mu, sigma = torch.chunk(output, chunks=2, dim=-1)
    assert mu.shape == batch.y.shape
    assert sigma.shape == batch.y.shape
    assert torch.all(sigma > 0.0)


def test_causal_cross_asset_transformer_accepts_optional_streams() -> None:
    """
    Run the model when only the continuous stream is present.

    Returns
    -------
    None
        This test asserts optional-stream support.
    """

    batch = PanelBatch(
        X_continuous=torch.randn(1, 2, 3, 2, dtype=torch.float32),
        X_continuous_mask=torch.ones(1, 2, 3, 2, dtype=torch.bool),
        X_cat_continuous=None,
        X_cat_continuous_mask=None,
        X_cat_static=None,
        X_cat_static_mask=None,
        y=torch.zeros(1, 2, 1, 1, dtype=torch.float32),
        y_mask=torch.ones(1, 2, 1, 1, dtype=torch.bool),
        id_index=["A", "B"],
        forecast_start_dates=[np.datetime64("2024-01-03")],
        forecast_end_dates=[np.datetime64("2024-01-03")],
    )
    config = CausalCrossAssetTransformerConfig(
        d_model=8,
        dim_feedforward=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        fusion_mode="sum",
        out_steps=1,
        target_dim=1,
    )
    model = CausalCrossAssetTransformer(config, continuous_dim=2)

    output = model(batch)
    assert output.shape == torch.Size([1, 2, 1, 2])


def test_model_factory_derives_categorical_cardinalities_from_preprocessing() -> None:
    """
    Build the transformer using categorical vocabulary sizes from preprocessing.

    Returns
    -------
    None
        This test asserts artifact-driven model construction.
    """

    panel = pd.DataFrame(
        {
            "asset_id": ["SPY", "SPY", "QQQ", "QQQ"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
            "return": [0.1, 0.2, 0.3, 0.4],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
        }
    )
    preprocessing = PreprocessingBundle().fit(
        panel,
        continuous_columns=("return",),
        categorical_columns=("ticker",),
    )
    model = build_causal_cross_asset_transformer(
        model_config=ModelConfig(
            name="cross_attention_volatility",
            model_dim=8,
            ff_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            fusion_mode="concat",
        ),
        feature_config=FeatureConfig(
            sequence_length=2,
            forecast_horizon=1,
        ),
        preprocessing=preprocessing,
    )

    assert model.continuous_dim == 1
    assert model.dynamic_categorical_dim is None
    assert model.static_categorical_dim == 1
    assert len(model.static_embeddings or []) == 1
    assert model.config.static_categorical_cardinalities == (4,)


def test_model_factory_builds_temporal_only_transformer() -> None:
    """
    Build the temporal-only transformer from the public model name.

    Returns
    -------
    None
        This test asserts model-name dispatch.
    """

    panel = pd.DataFrame(
        {
            "asset_id": ["SPY", "SPY", "QQQ", "QQQ"],
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
            "return": [0.1, 0.2, 0.3, 0.4],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
        }
    )
    preprocessing = PreprocessingBundle().fit(
        panel,
        continuous_columns=("return",),
        categorical_columns=("ticker",),
    )

    model = build_panel_model(
        model_config=ModelConfig(
            name="temporal_only_transformer",
            model_dim=8,
            ff_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            fusion_mode="concat",
            likelihood_eps=1e-5,
        ),
        feature_config=FeatureConfig(
            sequence_length=2,
            forecast_horizon=1,
        ),
        preprocessing=preprocessing,
    )

    batch = PanelBatch(
        X_continuous=torch.randn(2, 3, 4, 1, dtype=torch.float32),
        X_continuous_mask=torch.ones(2, 3, 4, 1, dtype=torch.bool),
        X_cat_continuous=None,
        X_cat_continuous_mask=None,
        X_cat_static=torch.zeros(2, 3, 4, 1, dtype=torch.long),
        X_cat_static_mask=torch.ones(2, 3, 4, 1, dtype=torch.bool),
        y=torch.zeros(2, 3, 1, 1, dtype=torch.float32),
        y_mask=torch.ones(2, 3, 1, 1, dtype=torch.bool),
        id_index=["A", "B", "C"],
        forecast_start_dates=[np.datetime64("2024-01-03"), np.datetime64("2024-01-04")],
        forecast_end_dates=[np.datetime64("2024-01-03"), np.datetime64("2024-01-04")],
    )

    output = model(batch)
    assert output.shape == torch.Size([2, 3, 1, 2])
