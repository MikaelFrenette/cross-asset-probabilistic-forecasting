"""
Shared Configuration
---------------------
Pydantic config schema, YAML loader, component builders, and logging setup
for the dlecosys pipeline.

Classes
-------
PipelineConfig
    Top-level validated config object produced by load_config.
LoggingSection
    Logging verbosity and format settings.

Functions
---------
load_config
    Load a YAML pipeline config with optional single-level base inheritance.
configure_logging
    Apply global logging configuration from a LoggingSection at startup.
build_optimizer
    Instantiate an optimizer from OptimizerSection.
build_loss
    Instantiate a loss function from a loss name string.
build_metrics
    Build the metrics dict from a list of metric name strings.
build_callbacks
    Build the callback list from CallbacksSection and a checkpoint path.
"""

from dlecosys.shared.config.builders import (
    build_callbacks,
    build_loss,
    build_metrics,
    build_optimizer,
    build_scheduler,
)
from dlecosys.shared.config.loader import load_config
from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.schema import (
    CallbacksSection,
    CheckpointSection,
    DataSection,
    EarlyStoppingSection,
    ExperimentSection,
    InferenceSection,
    LoggingSection,
    ModelSection,
    OptimizerSection,
    PipelineConfig,
    TrainingSection,
)

__all__ = [
    "PipelineConfig",
    "ExperimentSection",
    "DataSection",
    "ModelSection",
    "OptimizerSection",
    "EarlyStoppingSection",
    "CheckpointSection",
    "CallbacksSection",
    "TrainingSection",
    "InferenceSection",
    "LoggingSection",
    "load_config",
    "configure_logging",
    "build_optimizer",
    "build_loss",
    "build_metrics",
    "build_callbacks",
    "build_scheduler",
]
