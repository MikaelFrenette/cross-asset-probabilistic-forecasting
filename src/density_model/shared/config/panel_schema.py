"""
Panel Pipeline Configuration
----------------------------
Pydantic configuration objects for panel forecasting runs (training + inference).
These mirror the legacy density_model config shape while taking raw column
names directly (no public/internal alias indirection).

Classes
-------
PanelDataConfig
    Training-time dataset schema: train / validation CSV paths plus shared
    date and id columns.

PanelPredictDataConfig
    Inference-time dataset schema: input CSV path plus shared date / id columns.

PanelFeaturesConfig
    Feature selection and forecasting-window shape (sequence_length,
    forecast_horizon, target_column, continuous / categorical column lists).

PanelModelSpec
    Model registry lookup key plus a free-form params mapping forwarded to the
    registered model's Pydantic config class via ``ModelFactory.build``. The
    pipeline fills runtime-derived fields (``continuous_dim``,
    ``*_cardinalities``) before calling the factory.

PanelTrainingRuntimeConfig
    Optimization / training runtime knobs.

PanelEarlyStoppingConfig, PanelModelCheckpointConfig
    Callback config wrappers.

PanelArtifactsConfig
    Output directory and artifact filename conventions.

PanelLoggingConfig
    Standalone logging configuration (mirrors density_model ``LoggingConfig``).

PanelPipelineConfig
    Top-level training config container.

PanelPredictConfig
    Top-level inference config container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "PanelArtifactsConfig",
    "PanelDataConfig",
    "PanelDistributedConfig",
    "PanelEarlyStoppingConfig",
    "PanelEnsembleConfig",
    "PanelEnsemblePruningConfig",
    "PanelFeatureBootstrapperConfig",
    "PanelFeaturesConfig",
    "PanelLoggingConfig",
    "PanelModelCheckpointConfig",
    "PanelModelSpec",
    "PanelPipelineConfig",
    "PanelPredictConfig",
    "PanelPredictDataConfig",
    "PanelPrunerConfig",
    "PanelSampleBootstrapperConfig",
    "PanelSplitterConfig",
    "PanelTrainingRuntimeConfig",
    "PanelTuningConfig",
    "load_panel_pipeline_config",
    "load_panel_predict_config",
]


def _validate_non_empty(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(
            "string fields must be non-empty and must not include surrounding whitespace."
        )
    return value


class PanelDataConfig(BaseModel):
    """Training-time dataset schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_csv: str
    val_csv: str
    date_column: str = "date"
    id_column: str = "asset_id"

    @field_validator("train_csv", "val_csv", "date_column", "id_column")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)

    def train_csv_path(self) -> Path:
        return Path(self.train_csv)

    def val_csv_path(self) -> Path:
        return Path(self.val_csv)


class PanelPredictDataConfig(BaseModel):
    """Inference-time dataset schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    csv_path: str
    date_column: str = "date"
    id_column: str = "asset_id"

    @field_validator("csv_path", "date_column", "id_column")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)

    def csv_file(self) -> Path:
        return Path(self.csv_path)


class PanelFeaturesConfig(BaseModel):
    """Feature selection and forecasting-window shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_length: int = Field(gt=0)
    forecast_horizon: int = Field(gt=0)
    target_column: str
    continuous_columns: tuple[str, ...]
    dynamic_categorical_columns: tuple[str, ...] = ()
    static_categorical_columns: tuple[str, ...] = ()

    @field_validator("target_column")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)

    @field_validator("continuous_columns")
    @classmethod
    def _validate_continuous(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("At least one continuous column is required.")
        return tuple(_validate_non_empty(v) for v in value)

    @field_validator("dynamic_categorical_columns", "static_categorical_columns")
    @classmethod
    def _validate_categoricals(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validate_non_empty(v) for v in value)


class PanelModelSpec(BaseModel):
    """Registered model lookup key plus a params mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)


class PanelTrainingRuntimeConfig(BaseModel):
    """Optimization + training runtime knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    max_epochs: int = Field(gt=0)
    random_seed: int = 42
    device: str = "cpu"
    weight_decay: float = Field(default=0.0, ge=0.0)
    shuffle: bool = True
    grad_clip: float | None = Field(default=1.0, gt=0.0)
    verbose: int = Field(default=2, ge=0, le=2)


class PanelEarlyStoppingConfig(BaseModel):
    """Early-stopping callback configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = Field(default="min", pattern="^(min|max)$")
    patience: int = Field(default=10, ge=0)
    min_delta: float = 0.0
    warmup: int = Field(default=0, ge=0)
    restore_best_weights: bool = True
    verbose: bool = False


class PanelModelCheckpointConfig(BaseModel):
    """Model-checkpoint callback configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = Field(default="min", pattern="^(min|max)$")
    min_delta: float = 0.0
    warmup: int = Field(default=0, ge=0)
    save_optimizer: bool = True
    overwrite: bool = True
    verbose: bool = True


class PanelArtifactsConfig(BaseModel):
    """Output directory and artifact filename conventions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: str = "outputs/density_model"
    preprocessing_filename: str = "preprocessing.json"
    model_state_filename: str = "model.pt"
    manifest_filename: str = "training_manifest.json"
    predictions_filename: str = "predictions.csv"

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def preprocessing_path(self) -> Path:
        return self.output_path() / self.preprocessing_filename

    def model_state_path(self) -> Path:
        return self.output_path() / self.model_state_filename

    def manifest_path(self) -> Path:
        return self.output_path() / self.manifest_filename

    def predictions_path(self) -> Path:
        return self.output_path() / self.predictions_filename


class PanelLoggingConfig(BaseModel):
    """Validated logging configuration (matches density_model ``LoggingConfig``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    level: str = "INFO"
    format: str = "text"
    include_timestamps: bool = True


class PanelDistributedConfig(BaseModel):
    """DistributedDataParallel configuration for panel training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    backend: str = Field(default="nccl", pattern="^(nccl|gloo)$")


class PanelSplitterConfig(BaseModel):
    """Panel time-series CV splitter configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(default="expanding_window", pattern="^(holdout|expanding_window|walk_forward)$")
    n_splits: int = Field(default=5, ge=1)
    val_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    train_window_size: int | None = None


class PanelPrunerConfig(BaseModel):
    """Optuna pruner configuration for panel tuning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    type: str = Field(default="median", pattern="^(median|none)$")
    n_warmup_steps: int = Field(default=5, ge=0)
    n_startup_trials: int = Field(default=5, ge=0)


class PanelTuningConfig(BaseModel):
    """Hyperparameter tuning configuration for panel forecasting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_name: str
    direction: str = Field(default="minimize", pattern="^(minimize|maximize)$")
    metric: str = "val_loss"
    sampler: str = Field(default="grid", pattern="^(grid|random)$")
    n_trials: int | None = None
    epochs_per_fold: int = Field(default=10, gt=0)
    pruner: PanelPrunerConfig = Field(default_factory=PanelPrunerConfig)
    splitter: PanelSplitterConfig = Field(default_factory=PanelSplitterConfig)
    search_space: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("study_name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)


class PanelSampleBootstrapperConfig(BaseModel):
    """Sample-axis bootstrapper (bootstraps forecast dates) configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(default="with_replacement", pattern="^(with_replacement|no_bootstrap)$")
    max_samples: float = Field(default=1.0, gt=0.0, le=1.0)


class PanelFeatureBootstrapperConfig(BaseModel):
    """Feature-axis bootstrapper (bootstraps asset IDs) configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(default="all", pattern="^(all|random_subspace)$")
    max_features: float = Field(default=1.0, gt=0.0, le=1.0)


class PanelEnsemblePruningConfig(BaseModel):
    """Top-N pruning strategy for the bagging ensemble."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    keep: int = Field(default=10, ge=1)
    strategy: str = Field(default="top_n", pattern="^(top_n)$")


class PanelEnsembleConfig(BaseModel):
    """Bagging ensemble configuration for panel forecasting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(default="bagging", pattern="^(bagging)$")
    n_estimators: int = Field(default=10, ge=1)
    aggregation: str = Field(default="mean", pattern="^(mean|median)$")
    sample_bootstrapper: PanelSampleBootstrapperConfig = Field(
        default_factory=PanelSampleBootstrapperConfig
    )
    feature_bootstrapper: PanelFeatureBootstrapperConfig = Field(
        default_factory=PanelFeatureBootstrapperConfig
    )
    pruning: PanelEnsemblePruningConfig = Field(default_factory=PanelEnsemblePruningConfig)
    seed: int = 42


class PanelPipelineConfig(BaseModel):
    """Top-level panel training pipeline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: PanelDataConfig
    features: PanelFeaturesConfig
    model: PanelModelSpec
    training: PanelTrainingRuntimeConfig
    early_stopping: PanelEarlyStoppingConfig = Field(default_factory=PanelEarlyStoppingConfig)
    model_checkpoint: PanelModelCheckpointConfig = Field(
        default_factory=PanelModelCheckpointConfig
    )
    artifacts: PanelArtifactsConfig = Field(default_factory=PanelArtifactsConfig)
    logging: PanelLoggingConfig = Field(default_factory=PanelLoggingConfig)
    distributed: PanelDistributedConfig = Field(default_factory=PanelDistributedConfig)
    tuning: PanelTuningConfig | None = None
    ensemble: PanelEnsembleConfig | None = None


class PanelPredictConfig(BaseModel):
    """Top-level panel inference configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: str
    data: PanelPredictDataConfig
    output_csv: str
    device: str = "cpu"
    batch_size: int = Field(default=256, gt=0)
    plots_dir: str | None = None
    logging: PanelLoggingConfig = Field(default_factory=PanelLoggingConfig)

    @field_validator("manifest_path", "output_csv", "device")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)

    def manifest_file(self) -> Path:
        return Path(self.manifest_path)

    def output_path(self) -> Path:
        return Path(self.output_csv)

    def plots_path(self) -> Path:
        """Return the plots output directory (defaults to predictions CSV's folder + /plots)."""

        if self.plots_dir:
            return Path(self.plots_dir)
        return self.output_path().parent / "plots"


def load_panel_pipeline_config(path: str | Path) -> PanelPipelineConfig:
    """Load a YAML panel pipeline config and validate it."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Configuration file {path!r} is empty.")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Configuration file {path!r} must contain a top-level mapping."
        )
    return PanelPipelineConfig.model_validate(payload)


def load_panel_predict_config(path: str | Path) -> PanelPredictConfig:
    """Load a YAML panel prediction config and validate it."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Configuration file {path!r} is empty.")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Configuration file {path!r} must contain a top-level mapping."
        )
    return PanelPredictConfig.model_validate(payload)
