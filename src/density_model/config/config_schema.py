"""
Main Configuration
------------------
Shared typed configuration objects used by the public extraction, training,
prediction, and evaluation workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from density_model.data.yahoo import YahooDailyReturnsRequest

__all__ = [
    "ArtifactConfig",
    "DataConfig",
    "FeatureConfig",
    "FeatureStreamConfig",
    "InferenceConfig",
    "ModelConfig",
    "TrainingConfig",
]

ModelName = Literal["cross_attention_volatility", "causal_cross_asset_transformer", "temporal_only_transformer"]


class FeatureStreamConfig(BaseModel):
    """
    Configuration for assigning public features to model input streams.

    Parameters
    ----------
    continuous : tuple of str
        Public continuous features passed to ``X_continuous``.
    dynamic_categoricals : tuple of str or None, default=None
        Public dynamic categorical features passed to ``X_cat_continuous``.
    static_categoricals : tuple of str or None, default=None
        Public static categorical features passed to ``X_cat_static``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuous: tuple[str, ...]
    dynamic_categoricals: tuple[str | None, ...] | None = None
    static_categoricals: tuple[str | None, ...] | None = None

    @field_validator("continuous")
    @classmethod
    def validate_continuous(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """
        Validate configured continuous feature names.

        Parameters
        ----------
        value : tuple of str
            Raw continuous feature names.

        Returns
        -------
        tuple of str
            Validated continuous feature names.
        """

        if not value:
            raise ValueError("At least one continuous feature is required.")
        if any(not feature for feature in value):
            raise ValueError("Continuous feature names must be non-empty strings.")
        return value

    @field_validator("dynamic_categoricals", "static_categoricals")
    @classmethod
    def normalize_optional_feature_list(
        cls,
        value: tuple[str | None, ...] | None,
    ) -> tuple[str, ...] | None:
        """
        Normalize optional categorical feature lists by removing ``None`` entries.

        Parameters
        ----------
        value : tuple of str or None, or None
            Raw optional categorical feature names.

        Returns
        -------
        tuple of str or None
            Normalized categorical feature names, or ``None`` when no features are configured.
        """

        if value is None:
            return None
        normalized = tuple(feature for feature in value if feature is not None)
        if any(not feature for feature in normalized):
            raise ValueError("Categorical feature names must be non-empty strings when provided.")
        return normalized or None

    def resolve_continuous_columns(self) -> tuple[str, ...]:
        """
        Resolve public continuous feature names to internal panel column names.

        Returns
        -------
        tuple of str
            Internal panel column names for continuous features.
        """

        mapping = {
            "returns": "return",
        }
        return tuple(self._resolve_feature_name(feature=name, mapping=mapping, stream_name="continuous") for name in self.continuous)

    def resolve_dynamic_categorical_columns(self) -> tuple[str, ...]:
        """
        Resolve public dynamic categorical feature names to internal panel column names.

        Returns
        -------
        tuple of str
            Internal panel column names for dynamic categorical features.
        """

        mapping: dict[str, str] = {}
        names = self.dynamic_categoricals or ()
        return tuple(
            self._resolve_feature_name(feature=name, mapping=mapping, stream_name="dynamic_categoricals") for name in names
        )

    def resolve_static_categorical_columns(self) -> tuple[str, ...]:
        """
        Resolve public static categorical feature names to internal panel column names.

        Returns
        -------
        tuple of str
            Internal panel column names for static categorical features.
        """

        mapping = {
            "ticker": "ticker",
        }
        names = self.static_categoricals or ()
        return tuple(
            self._resolve_feature_name(feature=name, mapping=mapping, stream_name="static_categoricals") for name in names
        )

    def _resolve_feature_name(self, *, feature: str, mapping: dict[str, str], stream_name: str) -> str:
        """
        Resolve one public feature name to an internal panel column name.

        Parameters
        ----------
        feature : str
            Public feature name.
        mapping : dict of str to str
            Supported feature mapping for the stream.
        stream_name : str
            Human-readable stream name for error messages.

        Returns
        -------
        str
            Internal panel column name.

        Raises
        ------
        ValueError
            If the feature name is unsupported for the stream.
        """

        if feature not in mapping:
            raise ValueError(f"Unsupported feature {feature!r} configured for {stream_name}.")
        return mapping[feature]


class DataConfig(BaseModel):
    """
    Configuration for market data ingestion.

    Parameters
    ----------
    source : str
        Data provider identifier used by the pipeline.
    tickers : tuple of str
        Ticker symbols to download from the configured provider.
    start_date : str
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date : str
        Exclusive end date in ``YYYY-MM-DD`` format.
    price_field : str, default="Adj Close"
        Price column used when transforming prices into returns.
    calendar : str, default="XNYS"
        Master market calendar used for session normalization.
    drop_missing : bool, default=False
        Whether rows containing missing return values should be dropped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(pattern="^yahoo_finance$")
    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    price_field: str = "Adj Close"
    calendar: str = "XNYS"
    drop_missing: bool = False

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """
        Validate the configured ticker sequence.

        Parameters
        ----------
        value : tuple of str
            Raw ticker sequence.

        Returns
        -------
        tuple of str
            Validated ticker sequence.
        """

        if not value:
            raise ValueError("At least one ticker symbol is required.")
        return value

    def to_yahoo_request(self) -> YahooDailyReturnsRequest:
        """
        Convert the data configuration to a Yahoo Finance request.

        Returns
        -------
        YahooDailyReturnsRequest
            Request object that can be passed to the Yahoo Finance loader.

        Raises
        ------
        ValueError
            If the configured data source is not Yahoo Finance.
        """

        if self.source != "yahoo_finance":
            raise ValueError("YahooDailyReturnsRequest is only available for source='yahoo_finance'.")
        return YahooDailyReturnsRequest(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
            price_field=self.price_field,
            calendar=self.calendar,
            drop_missing=self.drop_missing,
        )


class FeatureConfig(BaseModel):
    """
    Configuration for feature construction from return series.

    Parameters
    ----------
    sequence_length : int
        Number of historical timesteps provided to the model.
    forecast_horizon : int
        Number of steps ahead used for the prediction target.
    target_column : str, default="return"
        Supervised target column predicted by the model.
    streams : FeatureStreamConfig
        Input-stream assignment for continuous and categorical model features.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_length: int = Field(gt=0)
    forecast_horizon: int = Field(gt=0)
    target_column: str = "return"
    streams: FeatureStreamConfig = Field(
        default_factory=lambda: FeatureStreamConfig(
            continuous=("returns",),
            dynamic_categoricals=None,
            static_categoricals=("ticker",),
        )
    )


class ModelConfig(BaseModel):
    """
    Configuration for the volatility model architecture.

    Parameters
    ----------
    name : {"cross_attention_volatility", "causal_cross_asset_transformer", "temporal_only_transformer"}
        Public model identifier.
    model_dim : int
        Shared transformer width used by the model.
    ff_dim : int
        Feed-forward width used inside transformer layers.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of stacked model layers.
    dropout : float
        Dropout rate used during training.
    fusion_mode : {"sum", "concat"}, default="concat"
        Stream-fusion strategy used before temporal encoding.
    likelihood_eps : float, default=1e-6
        Minimum positive offset added to the softplus sigma head for numerical stability.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ModelName
    model_dim: int = Field(gt=0)
    ff_dim: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_layers: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    fusion_mode: Literal["sum", "concat"] = "concat"
    likelihood_eps: float = Field(default=1e-6, gt=0.0)


class InferenceConfig(BaseModel):
    """
    Configuration for post-training prediction datasets.

    Parameters
    ----------
    start_date : str, default="2021-01-01"
        Earliest prediction-period date expected in external inference datasets.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_date: str = "2021-01-01"


class ArtifactConfig(BaseModel):
    """
    Configuration for saved training artifacts and prediction outputs.

    Parameters
    ----------
    output_dir : str, default="artifacts/demo"
        Directory where the training run stores model and preprocessing artifacts.
    preprocessing_filename : str, default="preprocessing.json"
        Filename used for the saved preprocessing bundle.
    model_state_filename : str, default="model.pt"
        Filename used for the saved torch model state dictionary.
    manifest_filename : str, default="training_manifest.json"
        Filename used for the frozen training manifest consumed by prediction.
    predictions_filename : str, default="predictions.csv"
        Default filename used for prediction outputs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: str = "artifacts/demo"
    preprocessing_filename: str = "preprocessing.json"
    model_state_filename: str = "model.pt"
    manifest_filename: str = "training_manifest.json"
    predictions_filename: str = "predictions.csv"

    def output_path(self) -> Path:
        """
        Return the configured artifact output directory.

        Returns
        -------
        pathlib.Path
            Output directory for saved artifacts.
        """

        return Path(self.output_dir)

    def preprocessing_path(self) -> Path:
        """
        Return the configured preprocessing artifact path.

        Returns
        -------
        pathlib.Path
            Full path to the preprocessing artifact.
        """

        return self.output_path() / self.preprocessing_filename

    def model_state_path(self) -> Path:
        """
        Return the configured model state path.

        Returns
        -------
        pathlib.Path
            Full path to the saved torch model state dictionary.
        """

        return self.output_path() / self.model_state_filename

    def manifest_path(self) -> Path:
        """
        Return the configured training manifest path.

        Returns
        -------
        pathlib.Path
            Full path to the frozen training manifest.
        """

        return self.output_path() / self.manifest_filename

    def predictions_path(self) -> Path:
        """
        Return the configured predictions output path.

        Returns
        -------
        pathlib.Path
            Full path to the default prediction output file.
        """

        return self.output_path() / self.predictions_filename


class TrainingConfig(BaseModel):
    """
    Configuration for optimization and training behavior.

    Parameters
    ----------
    batch_size : int
        Number of samples per optimization step.
    learning_rate : float
        Optimizer learning rate.
    max_epochs : int
        Maximum number of training epochs.
    random_seed : int
        Seed used to initialize deterministic training components.
    device : str, default="cpu"
        Torch device passed to the trainer and model runtime.
    weight_decay : float, default=0.0
        Optimizer weight decay.
    shuffle : bool, default=True
        Whether to shuffle the training dataloader.
    grad_clip : float or None, default=None
        Maximum gradient norm. ``None`` disables clipping.
    verbose : int, default=2
        Trainer verbosity level.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    max_epochs: int = Field(gt=0)
    random_seed: int
    device: str = "cpu"
    weight_decay: float = Field(default=0.0, ge=0.0)
    shuffle: bool = True
    grad_clip: float | None = Field(default=None, gt=0.0)
    verbose: int = Field(default=2, ge=0, le=2)
