"""
Pipeline Component Builders
----------------------------
Factory functions that construct PyTorch / training objects from config sections.

Functions
---------
build_optimizer
    Instantiate an optimizer from OptimizerSection.
build_loss
    Instantiate a loss function from a loss name string.
build_metrics
    Build the metrics dict from a list of metric name strings.
build_scheduler
    Instantiate a torch.optim.lr_scheduler from SchedulerSection + an optimizer.
build_callbacks
    Build the callback list from CallbacksSection, paths, and (optionally) a scheduler.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from dlecosys.shared.config.schema import CallbacksSection, OptimizerSection, SchedulerSection
from dlecosys.shared.metrics import accuracy, binary_accuracy, mae, mse, rmse
from dlecosys.shared.training.callbacks import (
    EarlyStopping,
    LRSchedulerCallback,
    ModelCheckpoint,
    TensorBoardCallback,
)

__all__ = [
    "build_callbacks",
    "build_loss",
    "build_metrics",
    "build_optimizer",
    "build_scheduler",
]

_OPTIMIZERS = {
    "adam": lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
    "sgd": lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd),
    "adamw": lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
}

_LOSSES: Dict[str, type] = {
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
}

_METRICS: Dict[str, Callable] = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "accuracy": accuracy,
    "binary_accuracy": binary_accuracy,
}

_SCHEDULERS: Dict[str, type] = {
    "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def build_optimizer(model: nn.Module, cfg: OptimizerSection) -> Optimizer:
    """Build an optimizer from config."""
    builder = _OPTIMIZERS[cfg.name]
    return builder(model.parameters(), cfg.lr, cfg.weight_decay)


def build_loss(name: str) -> nn.Module:
    """Instantiate a loss function by name."""
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss {name!r}. Available: {sorted(_LOSSES)}")
    return _LOSSES[name]()


def build_metrics(names: List[str]) -> Dict[str, Callable[..., Any]]:
    """Build the metrics dict from a list of metric name strings."""
    unknown = [n for n in names if n not in _METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}. Available: {sorted(_METRICS)}")
    return {name: _METRICS[name] for name in names}


def build_scheduler(cfg: SchedulerSection, optimizer: Optimizer):
    """Instantiate a torch LR scheduler from config. ``cfg.params`` is forwarded as kwargs."""
    scheduler_cls = _SCHEDULERS[cfg.type]
    return scheduler_cls(optimizer, **cfg.params)


def build_callbacks(
    cfg: CallbacksSection,
    *,
    checkpoint_path: str,
    tensorboard_dir: Optional[str] = None,
) -> list:
    """Build the standard callback list from CallbacksSection.

    Parameters
    ----------
    cfg : CallbacksSection
    checkpoint_path : str
        Path where ModelCheckpoint should write.
    tensorboard_dir : str, optional
        Directory for TensorBoard event files. Required when
        ``cfg.tensorboard.enabled`` is True.
    """
    callbacks = []
    es = cfg.early_stopping
    if es.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=es.monitor,
                patience=es.patience,
                mode=es.mode,
                warmup=es.warmup,
            )
        )
    ck = cfg.checkpoint
    if ck.enabled:
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=ck.monitor,
                verbose=False,
            )
        )
    tb = cfg.tensorboard
    if tb.enabled:
        if tensorboard_dir is None:
            raise ValueError(
                "tensorboard.enabled=true but no tensorboard_dir was provided to build_callbacks"
            )
        callbacks.append(TensorBoardCallback(log_dir=tensorboard_dir))
    return callbacks
