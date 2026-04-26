"""
Shared Training Infrastructure
------------------------------
Trainer abstractions used by the panel pipeline:

- :class:`BaseTrainer` — abstract loop coordinator (callbacks, metrics, progress)
- :class:`Callback` / :class:`CallbackList` — lifecycle hook protocol
- :class:`EarlyStopping`, :class:`ModelCheckpoint`,
  :class:`LRSchedulerCallback`, :class:`GradNormCallback`,
  :class:`TensorBoardCallback` — concrete callbacks
- :class:`MetricsTracker`, :class:`ProgressBar` — instrumentation utilities

Panel-specific concrete trainers, runners, losses, and manifests live in
the submodule files (``panel_trainer``, ``panel_distributed``,
``panel_losses``, ``panel_run``, ``panel_preprocess``,
``panel_manifest``); import directly from there.
"""

from density_model.shared.training.base import BaseTrainer
from density_model.shared.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    GradNormCallback,
    LRSchedulerCallback,
    ModelCheckpoint,
    TensorBoardCallback,
)
from density_model.shared.training.utils import MetricsTracker, ProgressBar

__all__ = [
    "BaseTrainer",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "GradNormCallback",
    "LRSchedulerCallback",
    "ModelCheckpoint",
    "TensorBoardCallback",
    "MetricsTracker",
    "ProgressBar",
]
