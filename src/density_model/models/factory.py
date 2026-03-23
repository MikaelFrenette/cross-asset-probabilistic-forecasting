"""
Model Factory
-------------
Factories that derive model construction arguments from typed repository config
and fitted preprocessing artifacts.
"""

from __future__ import annotations

from density_model.config.config_schema import FeatureConfig, ModelConfig
from density_model.models.config import CausalCrossAssetTransformerConfig
from density_model.models.cross_asset import CausalCrossAssetTransformer
from density_model.models.temporal_only import TemporalOnlyTransformer
from density_model.preprocessing import PreprocessingBundle

__all__ = ["build_causal_cross_asset_transformer", "build_panel_model"]


def build_causal_cross_asset_transformer(
    *,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    preprocessing: PreprocessingBundle,
) -> CausalCrossAssetTransformer:
    """
    Build the repository's causal cross-asset transformer from fitted artifacts.

    Parameters
    ----------
    model_config : ModelConfig
        Typed repository model configuration.
    feature_config : FeatureConfig
        Typed feature configuration defining active model streams.
    preprocessing : PreprocessingBundle
        Fitted preprocessing bundle used to derive categorical vocabulary sizes.

    Returns
    -------
    CausalCrossAssetTransformer
        Model instance configured from repository settings and fitted artifacts.
    """

    continuous_columns = feature_config.streams.resolve_continuous_columns()
    dynamic_categorical_columns = feature_config.streams.resolve_dynamic_categorical_columns()
    static_categorical_columns = feature_config.streams.resolve_static_categorical_columns()
    tokenizer = preprocessing.categorical_tokenizer
    dynamic_cardinalities = None
    static_cardinalities = None
    if dynamic_categorical_columns or static_categorical_columns:
        if tokenizer is None:
            raise ValueError("Categorical streams require a fitted categorical tokenizer in preprocessing.")
        if dynamic_categorical_columns:
            dynamic_cardinalities = tokenizer.get_vocabulary_sizes(dynamic_categorical_columns)
        if static_categorical_columns:
            static_cardinalities = tokenizer.get_vocabulary_sizes(static_categorical_columns)

    transformer_config = CausalCrossAssetTransformerConfig(
        d_model=model_config.model_dim,
        dim_feedforward=model_config.ff_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        fusion_mode=model_config.fusion_mode,
        out_steps=1,
        target_dim=1,
        dynamic_categorical_cardinalities=dynamic_cardinalities,
        static_categorical_cardinalities=static_cardinalities,
    )
    return CausalCrossAssetTransformer(
        transformer_config,
        continuous_dim=len(continuous_columns) or None,
        dynamic_categorical_dim=len(dynamic_categorical_columns) or None,
        static_categorical_dim=len(static_categorical_columns) or None,
        sigma_min=model_config.likelihood_eps,
    )


def build_panel_model(
    *,
    model_config: ModelConfig,
    feature_config: FeatureConfig,
    preprocessing: PreprocessingBundle,
):
    """
    Build a panel forecasting model selected by the public model name.

    Parameters
    ----------
    model_config : ModelConfig
        Typed repository model configuration.
    feature_config : FeatureConfig
        Typed feature configuration defining active model streams.
    preprocessing : PreprocessingBundle
        Fitted preprocessing bundle used to derive categorical vocabulary sizes.

    Returns
    -------
    object
        Constructed panel forecasting model.
    """

    if model_config.name in {"cross_attention_volatility", "causal_cross_asset_transformer"}:
        return build_causal_cross_asset_transformer(
            model_config=model_config,
            feature_config=feature_config,
            preprocessing=preprocessing,
        )

    continuous_columns = feature_config.streams.resolve_continuous_columns()
    dynamic_categorical_columns = feature_config.streams.resolve_dynamic_categorical_columns()
    static_categorical_columns = feature_config.streams.resolve_static_categorical_columns()
    tokenizer = preprocessing.categorical_tokenizer
    dynamic_cardinalities = None
    static_cardinalities = None
    if dynamic_categorical_columns or static_categorical_columns:
        if tokenizer is None:
            raise ValueError("Categorical streams require a fitted categorical tokenizer in preprocessing.")
        if dynamic_categorical_columns:
            dynamic_cardinalities = tokenizer.get_vocabulary_sizes(dynamic_categorical_columns)
        if static_categorical_columns:
            static_cardinalities = tokenizer.get_vocabulary_sizes(static_categorical_columns)

    transformer_config = CausalCrossAssetTransformerConfig(
        d_model=model_config.model_dim,
        dim_feedforward=model_config.ff_dim,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        fusion_mode=model_config.fusion_mode,
        out_steps=1,
        target_dim=1,
        dynamic_categorical_cardinalities=dynamic_cardinalities,
        static_categorical_cardinalities=static_cardinalities,
    )
    if model_config.name == "temporal_only_transformer":
        return TemporalOnlyTransformer(
            transformer_config,
            continuous_dim=len(continuous_columns) or None,
            dynamic_categorical_dim=len(dynamic_categorical_columns) or None,
            static_categorical_dim=len(static_categorical_columns) or None,
            sigma_min=model_config.likelihood_eps,
        )
    raise ValueError(f"Unsupported model name {model_config.name!r}.")
