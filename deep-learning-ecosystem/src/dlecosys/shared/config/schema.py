"""
Pipeline Config Schema
-----------------------
Pydantic models for the full dlecosys pipeline configuration tree.

Classes
-------
ExperimentSection
    Run identity: name, run_id, seed, output directory.
DataSection
    Synthetic data generation parameters and split sizes.
ModelSection
    Model registry key and parameter dict forwarded to ModelFactory.
OptimizerSection
    Optimizer name, learning rate, and weight decay.
EarlyStoppingSection
    EarlyStopping callback configuration.
CheckpointSection
    ModelCheckpoint callback configuration.
TensorBoardSection
    TensorBoard callback configuration.
CallbacksSection
    Container for all callback configs.
SchedulerSection
    Learning-rate scheduler configuration (type + params).
TrainingSection
    Full training loop configuration.
InferenceSection
    Inference / prediction configuration.
LoggingSection
    Logging verbosity and format settings.
DistributedSection
    Multi-GPU DDP configuration.
PrunerSection
    Optuna pruner configuration for tuning.
SplitterSection
    Validation splitter configuration for tuning (holdout / kfold / ...).
TuningSection
    Hyperparameter tuning configuration (sampler, pruner, splitter, search space).
EnsembleDistributedSection
    Per-estimator DDP configuration (schema only in v1; enabling raises).
SampleBootstrapperSection
    Row-level bootstrap scheme (with_replacement / no_bootstrap / user-defined).
FeatureBootstrapperSection
    Column-level bootstrap scheme (all / random_subspace / user-defined).
EnsemblePruningSection
    Post-training top-N selection of estimators by OOB metric.
EnsembleSection
    Bagging ensemble configuration (composes the two bootstrapper sections).
PipelineConfig
    Top-level config validated from a merged YAML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "PipelineConfig",
    "ExperimentSection",
    "DataSection",
    "ModelSection",
    "OptimizerSection",
    "EarlyStoppingSection",
    "CheckpointSection",
    "TensorBoardSection",
    "CallbacksSection",
    "SchedulerSection",
    "TrainingSection",
    "InferenceSection",
    "LoggingSection",
    "DistributedSection",
    "PrunerSection",
    "SplitterSection",
    "TuningSection",
    "EnsembleDistributedSection",
    "SampleBootstrapperSection",
    "FeatureBootstrapperSection",
    "EnsemblePruningSection",
    "EnsembleSection",
    "SyntheticSection",
    "SyntheticConfig",
]


class ExperimentSection(BaseModel):
    name: str
    seed: int = 42
    deterministic: bool = True
    output_dir: str = "outputs"


class DataSection(BaseModel):
    task: Literal["regression", "classification"]
    path: str
    target_col: str = "target"
    test_size: float = 0.2
    val_size: float = 0.1
    batch_size: int = 32
    scaling: Optional[Literal["standard", "minmax"]] = "standard"
    categorical_cols: List[int] = Field(default_factory=list)


class SyntheticSection(BaseModel):
    """Generation parameters for a single synthetic dataset."""

    task: Literal["regression", "classification"]
    n_samples: int = 1000
    n_features: int = 10
    n_classes: Optional[int] = None
    noise: float = 0.1
    seed: int = 42
    output_path: str
    target_col: str = "target"

    @model_validator(mode="after")
    def _classification_requires_n_classes(self) -> "SyntheticSection":
        if self.task == "classification" and self.n_classes is None:
            raise ValueError("n_classes is required when task is 'classification'")
        return self


class SyntheticConfig(BaseModel):
    """Top-level config for generate_synthetic.py."""

    regression: Optional[SyntheticSection] = None
    classification: Optional[SyntheticSection] = None


class ModelSection(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class OptimizerSection(BaseModel):
    name: Literal["adam", "sgd", "adamw"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0


class EarlyStoppingSection(BaseModel):
    enabled: bool = True
    monitor: str = "val_loss"
    patience: int = 10
    mode: Literal["min", "max"] = "min"
    warmup: int = 0


class CheckpointSection(BaseModel):
    enabled: bool = True
    monitor: str = "val_loss"


class TensorBoardSection(BaseModel):
    enabled: bool = False


class CallbacksSection(BaseModel):
    early_stopping: EarlyStoppingSection = Field(default_factory=EarlyStoppingSection)
    checkpoint: CheckpointSection = Field(default_factory=CheckpointSection)
    tensorboard: TensorBoardSection = Field(default_factory=TensorBoardSection)


class SchedulerSection(BaseModel):
    type: Literal[
        "cosine_annealing",
        "step",
        "exponential",
        "reduce_on_plateau",
    ]
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainingSection(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    epochs: int = 50
    loss: Literal["mse", "cross_entropy", "bce"] = "mse"
    optimizer: OptimizerSection = Field(default_factory=OptimizerSection)
    scheduler: Optional[SchedulerSection] = None
    metrics: List[str] = Field(default_factory=list)
    callbacks: CallbacksSection = Field(default_factory=CallbacksSection)
    amp: bool = False
    grad_accumulation_steps: int = 1
    grad_clip: float = 1.0
    verbose: int = 1

    @field_validator("grad_accumulation_steps")
    @classmethod
    def _positive_accum(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"grad_accumulation_steps must be >= 1; got {v}")
        return v


class InferenceSection(BaseModel):
    batch_size: int = 64


class LoggingSection(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["text"] = "text"
    include_timestamps: bool = True


class DistributedSection(BaseModel):
    enabled: bool = False
    backend: Literal["nccl", "gloo"] = "nccl"


class PrunerSection(BaseModel):
    enabled: bool = False
    type: Literal["median", "none"] = "median"
    n_warmup_steps: int = 5
    n_startup_trials: int = 5


class SplitterSection(BaseModel):
    type: Literal["holdout", "kfold", "stratified_kfold"] = "holdout"
    n_splits: int = 5
    shuffle: bool = True
    seed: int = 42


class TuningSection(BaseModel):
    study_name: str
    direction: Literal["minimize", "maximize"] = "minimize"
    metric: str = "val_loss"
    sampler: Literal["grid", "random"] = "grid"
    n_trials: Optional[int] = None
    pruner: PrunerSection = Field(default_factory=PrunerSection)
    splitter: SplitterSection = Field(default_factory=SplitterSection)
    scale_splits: bool = True
    storage: Optional[str] = None
    search_space: Dict[str, List[Any]]

    @field_validator("search_space")
    @classmethod
    def _non_empty_search_space(cls, v: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if not v:
            raise ValueError("search_space must contain at least one entry")
        for path, choices in v.items():
            if not choices:
                raise ValueError(f"search_space entry '{path}' has no choices")
        return v


class EnsembleDistributedSection(BaseModel):
    enabled: bool = False
    backend: Literal["nccl", "gloo"] = "nccl"


class SampleBootstrapperSection(BaseModel):
    type: Literal["with_replacement", "no_bootstrap"] = "with_replacement"
    max_samples: float = 1.0


class FeatureBootstrapperSection(BaseModel):
    type: Literal["all", "random_subspace"] = "all"
    max_features: float = 1.0


class EnsemblePruningSection(BaseModel):
    enabled: bool = False
    keep: int = 10
    strategy: Literal["top_n"] = "top_n"
    direction: Literal["minimize", "maximize"] = "minimize"


class EnsembleSection(BaseModel):
    type: Literal["bagging"] = "bagging"
    n_estimators: int = 20
    aggregation: Literal["mean", "median", "soft_vote", "majority"] = "mean"
    seed: int = 42
    sample_bootstrapper: SampleBootstrapperSection = Field(default_factory=SampleBootstrapperSection)
    feature_bootstrapper: FeatureBootstrapperSection = Field(default_factory=FeatureBootstrapperSection)
    pruning: EnsemblePruningSection = Field(default_factory=EnsemblePruningSection)
    distributed: EnsembleDistributedSection = Field(default_factory=EnsembleDistributedSection)


class PipelineConfig(BaseModel):
    experiment: ExperimentSection
    data: DataSection
    model: ModelSection
    training: TrainingSection
    inference: InferenceSection = Field(default_factory=InferenceSection)
    logging: LoggingSection = Field(default_factory=LoggingSection)
    distributed: DistributedSection = Field(default_factory=DistributedSection)
    tuning: Optional[TuningSection] = None
    ensemble: Optional[EnsembleSection] = None
