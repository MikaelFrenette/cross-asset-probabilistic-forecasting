"""
Train Configuration
-------------------
Typed configuration objects for CSV-based model training workflows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from density_model.config.config_schema import ArtifactConfig, FeatureConfig, ModelConfig, TrainingConfig

__all__ = [
    "CsvDatasetConfig",
    "EarlyStoppingWorkflowConfig",
    "ModelCheckpointWorkflowConfig",
    "TrainConfig",
    "TrainWorkflowDatasetConfig",
    "load_train_config",
]


class CsvDatasetConfig(BaseModel):
    """
    CSV dataset schema configuration for training and prediction workflows.

    Parameters
    ----------
    csv_path : str
        Path to the raw extracted feature CSV.
    date_column : str, default="date"
        Date column used for chronological normalization.
    id_column : str, default="asset_id"
        Entity identifier column used for panel construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    csv_path: str
    date_column: str = "date"
    id_column: str = "asset_id"

    @field_validator("csv_path", "date_column", "id_column")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """
        Validate required string fields for CSV dataset configs.

        Parameters
        ----------
        value : str
            Candidate string value.

        Returns
        -------
        str
            Validated string value.
        """

        if not value or value.strip() != value:
            raise ValueError("CSV dataset fields must be non-empty and must not include surrounding whitespace.")
        return value

    def csv_file(self) -> Path:
        """
        Return the configured CSV path.

        Returns
        -------
        pathlib.Path
            Dataset CSV path.
        """

        return Path(self.csv_path)


class TrainWorkflowDatasetConfig(BaseModel):
    """
    CSV dataset configuration for train/validation training runs.

    Parameters
    ----------
    train_csv : str
        Path to the extracted training feature CSV.
    validation_csv : str
        Path to the extracted validation feature CSV.
    date_column : str, default="date"
        Date column used for chronological normalization.
    id_column : str, default="asset_id"
        Entity identifier column used for panel construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    train_csv: str
    validation_csv: str
    date_column: str = "date"
    id_column: str = "asset_id"

    @field_validator("train_csv", "validation_csv", "date_column", "id_column")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """
        Validate required string fields for training dataset configs.

        Parameters
        ----------
        value : str
            Candidate string value.

        Returns
        -------
        str
            Validated string value.
        """

        if not value or value.strip() != value:
            raise ValueError("Training dataset fields must be non-empty and must not include surrounding whitespace.")
        return value

    def train_csv_file(self) -> Path:
        """
        Return the configured training CSV path.

        Returns
        -------
        pathlib.Path
            Training CSV path.
        """

        return Path(self.train_csv)

    def validation_csv_file(self) -> Path:
        """
        Return the configured validation CSV path.

        Returns
        -------
        pathlib.Path
            Validation CSV path.
        """

        return Path(self.validation_csv)


class EarlyStoppingWorkflowConfig(BaseModel):
    """
    Public training-config wrapper for early stopping.

    Parameters
    ----------
    enabled : bool, default=False
        Whether early stopping is active.
    monitor : str, default="val_loss"
        Metric name monitored for stopping decisions.
    mode : {"min", "max"}, default="min"
        Optimization direction for the monitored metric.
    patience : int, default=10
        Number of epochs without improvement before stopping.
    min_delta : float, default=0.0
        Minimum improvement threshold.
    warmup : int, default=0
        Number of initial epochs ignored by early stopping.
    restore_best_weights : bool, default=True
        Whether to restore the best model weights before exiting.
    verbose : bool, default=False
        Whether the callback should log status messages.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = Field(default="min", pattern="^(min|max)$")
    patience: int = Field(default=10, ge=0)
    min_delta: float = 0.0
    warmup: int = Field(default=0, ge=0)
    restore_best_weights: bool = True
    verbose: bool = False


class ModelCheckpointWorkflowConfig(BaseModel):
    """
    Public training-config wrapper for model checkpointing.

    Parameters
    ----------
    enabled : bool, default=False
        Whether model checkpointing is active.
    monitor : str, default="val_loss"
        Metric name monitored for checkpoint decisions.
    mode : {"min", "max"}, default="min"
        Optimization direction for the monitored metric.
    min_delta : float, default=0.0
        Minimum improvement threshold.
    warmup : int, default=0
        Number of initial epochs ignored by checkpointing.
    save_optimizer : bool, default=True
        Whether optimizer state is stored in the checkpoint file.
    overwrite : bool, default=True
        Whether to overwrite the previous best checkpoint.
    verbose : bool, default=True
        Whether the callback should log checkpoint saves.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    monitor: str = "val_loss"
    mode: str = Field(default="min", pattern="^(min|max)$")
    min_delta: float = 0.0
    warmup: int = Field(default=0, ge=0)
    save_optimizer: bool = True
    overwrite: bool = True
    verbose: bool = True


class TrainConfig(BaseModel):
    """
    Top-level configuration for CSV-based model training.

    Parameters
    ----------
    dataset : TrainWorkflowDatasetConfig
        Training and validation feature CSV paths plus shared schema information.
    features : FeatureConfig
        Feature engineering and stream configuration.
    model : ModelConfig
        Model architecture configuration.
    training : TrainingConfig
        Training runtime configuration.
    early_stopping : EarlyStoppingWorkflowConfig
        Early stopping callback configuration.
    model_checkpoint : ModelCheckpointWorkflowConfig
        Model checkpoint callback configuration.
    artifacts : ArtifactConfig
        Output artifact configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: TrainWorkflowDatasetConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    early_stopping: EarlyStoppingWorkflowConfig = Field(default_factory=EarlyStoppingWorkflowConfig)
    model_checkpoint: ModelCheckpointWorkflowConfig = Field(default_factory=ModelCheckpointWorkflowConfig)
    artifacts: ArtifactConfig


def load_train_config(path: str | Path) -> TrainConfig:
    """
    Load a training YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    TrainConfig
        Parsed training configuration.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(payload, dict):
        raise TypeError("The training configuration file must contain a top-level mapping.")
    return TrainConfig.model_validate(payload)
