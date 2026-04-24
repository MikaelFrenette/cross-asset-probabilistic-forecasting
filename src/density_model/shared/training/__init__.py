"""
Shared Training Infrastructure
--------------------------------
Reusable training loop abstractions for deep learning models.

Classes
-------
BaseTrainer
    Abstract trainer coordinating training/validation loops, callbacks,
    metric accumulation, and progress display.
Trainer
    Concrete supervised trainer with gradient clipping, AMP, and gradient
    accumulation.
DistributedTrainer
    Trainer subclass with DDP support for multi-GPU training.
Callback
    Base class for training lifecycle hooks.
CallbackList
    Container that dispatches lifecycle events to a list of callbacks.
EarlyStopping
    Callback that stops training when a monitored metric stops improving.
ModelCheckpoint
    Callback that saves model weights when a monitored metric improves.
LRSchedulerCallback
    Callback that steps a PyTorch LR scheduler at the end of each epoch.
GradNormCallback
    Callback that logs the total gradient L2 norm at each training step.
LRFinder
    Learning rate range test (Smith 2015).
LRFinderResult
    Result of a learning rate range test with suggestion helpers.
MetricsTracker
    Accumulates and snapshots training metrics.
ProgressBar
    Console progress bar for epoch/step-based training loops.
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
from density_model.shared.training.distributed import DistributedTrainer
from density_model.shared.training.lr_finder import LRFinder, LRFinderResult
from density_model.shared.training.run import build_dataloaders, run_training
from density_model.shared.training.supervised import Trainer
from density_model.shared.training.utils import MetricsTracker, ProgressBar

__all__ = [
    "BaseTrainer",
    "Trainer",
    "DistributedTrainer",
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedulerCallback",
    "GradNormCallback",
    "TensorBoardCallback",
    "LRFinder",
    "LRFinderResult",
    "MetricsTracker",
    "ProgressBar",
    "build_dataloaders",
    "run_training",
]
