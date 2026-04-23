"""
Training Base Classes
=====================
Abstract trainer base class for supervised deep learning loops.

Classes
-------
BaseTrainer
    Abstract base trainer coordinating training/validation loops, metric
    computation, progress display, and state logging. It validates inputs
    through :class:`config.trainer_config.BaseTrainerConfig`.

Notes
-----
- This class expects a configuration object produced by
  :class:`config.trainer_config.BaseTrainerConfig`.
- Callback infrastructure is defined in :mod:`training.callbacks`.
- Progress and metrics utilities are provided by :mod:`training.utils`.

Author  : Mikael Frenette
Contact : mikael_fr@hotmail.ca
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from ..config.trainer_config import BaseTrainerConfig
from ..training.callbacks import Callback, CallbackList
from ..training.utils import ProgressBar, MetricsTracker

__all__ = ["BaseTrainer"]


def _infer_batch_size(batch: Any) -> int:
    """Recursively extract batch size from the first tensor-like in the batch."""
    if isinstance(batch, (tuple, list)):
        return _infer_batch_size(batch[0])
    if hasattr(batch, "shape"):
        return int(batch.shape[0])
    if hasattr(batch, "__len__"):
        return len(batch)
    return 1


class BaseTrainer(ABC):
    """
    Abstract trainer coordinating training and validation loops.

    Parameters
    ----------
    **kwargs : dict
        Arguments forwarded to BaseTrainerConfig, including:
        - model : torch.nn.Module
        - optimizer : torch.optim.Optimizer
        - loss_fn : callable exposing forward(y_pred, y_true)
        - metrics : dict[str, callable]
        - callbacks : list of callback objects
        - verbose : int in {0,1,2}
        - strict : bool
        - device : torch.device or str, optional

    Attributes
    ----------
    cfg : BaseTrainerConfig
        Validated configuration.
    logger : MetricsTracker
        Aggregates and exposes running metrics.
    callbacks : CallbackList
        Container dispatching lifecycle hooks.
    stop_training : bool
        Flag settable by callbacks to end training early.

    Notes
    -----
    Subclasses must implement train_step and validation_step and return
    (metrics, info) where info includes 'y_true' and 'y_pred'.
    """

    def __init__(self, **kwargs):
        self.cfg = BaseTrainerConfig(**kwargs)
        if self.cfg.device is not None:
            self.cfg.model.to(self.cfg.device)
        self.logger = MetricsTracker()
        self.callbacks = CallbackList(self.cfg.callbacks or [])
        self.callbacks.set_trainer(self)
        self.stop_training: bool = False

    def train(self, *, train_dataloader, epochs: int, val_dataloader=None) -> None:
        """Orchestrate epochs: per-epoch callbacks, train phase, optional val phase, logger push."""
        self.logger.reset_state()

        try:
            steps_per_epoch = len(train_dataloader)
        except Exception:
            steps_per_epoch = 0

        progress_bar = ProgressBar(
            name="Training",
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            length=15,
            eta_smoothing=0.2,
        )

        self.callbacks.on_fit_start()

        try:
            for epoch in range(epochs):
                if self.stop_training:
                    break

                self.callbacks.on_epoch_start(epoch)
                self._on_train_epoch_start(epoch)

                self._train_epoch(epoch, train_dataloader, progress_bar)

                if val_dataloader is not None:
                    self._validate_epoch(epoch, val_dataloader, progress_bar)
                    progress_bar.reset(steps_per_epoch=steps_per_epoch, name="Training")
                elif self.verbose > 0:
                    progress_bar.end_epoch(epoch, self.logger.last_log())

                self.logger.push(epoch=epoch, step=steps_per_epoch if steps_per_epoch else None)
                self.callbacks.on_epoch_end(epoch, self.logger.last_log())

                if self.stop_training:
                    break

        except BaseException as e:
            self.callbacks.on_exception(e)
            raise
        finally:
            self.callbacks.on_fit_end()

    def _train_epoch(self, epoch: int, train_dataloader, progress_bar: ProgressBar) -> None:
        """Run one training epoch. Updates logger state and fires step callbacks."""
        declared_bs = getattr(train_dataloader, "batch_size", None)
        try:
            steps_per_epoch = len(train_dataloader)
        except Exception:
            steps_per_epoch = 0

        self.cfg.model.train()
        samples_seen = 0
        totals: Dict[str, float] = {}

        for step, batch in enumerate(train_dataloader, start=1):
            bs = _infer_batch_size(batch)
            if declared_bs is not None and bs < declared_bs:
                continue

            self.callbacks.on_train_step_start(step, batch)

            step_metrics, info = self._run_training_step(batch)
            step_metrics = {
                k: (float(v.item()) if hasattr(v, "item") else float(v))
                for k, v in step_metrics.items()
            }

            samples_seen += bs
            step_metrics.update(self._compute_metrics(info, prefix="train_"))

            for k, v in step_metrics.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            self.logger.update_state(**{k: totals[k] / samples_seen for k in totals})

            self.callbacks.on_train_step_end(
                step=step,
                batch=batch,
                outputs={"metrics": step_metrics, "info": info},
                logs=self.logger.last_log(),
            )

            if self.verbose == 2 and steps_per_epoch:
                progress_bar(epoch, step, self.logger.last_log())

    def _validate_epoch(self, epoch: int, val_dataloader, progress_bar: ProgressBar) -> None:
        """Run one validation epoch. Updates logger state and fires step callbacks."""
        try:
            val_steps = len(val_dataloader)
        except Exception:
            val_steps = 0

        if self.verbose == 2:
            progress_bar.end_epoch(epoch, self.logger.last_log())
        progress_bar.reset(steps_per_epoch=val_steps, name="Validation")

        self.cfg.model.eval()
        samples_seen = 0
        totals: Dict[str, float] = {}

        with torch.no_grad():
            for vstep, batch in enumerate(val_dataloader, start=1):
                self.callbacks.on_validation_step_start(vstep, batch)

                step_metrics, info = self._run_validation_step(batch)
                step_metrics = {
                    f"val_{k}": (float(v.item()) if hasattr(v, "item") else float(v))
                    for k, v in step_metrics.items()
                }

                bs = _infer_batch_size(batch)
                samples_seen += bs
                step_metrics.update(self._compute_metrics(info, prefix="val_"))

                for k, v in step_metrics.items():
                    totals[k] = totals.get(k, 0.0) + v * bs
                self.logger.update_state(**{k: totals[k] / samples_seen for k in totals})

                self.callbacks.on_validation_step_end(
                    vstep=vstep,
                    batch=batch,
                    outputs={"metrics": step_metrics, "info": info},
                    logs=self.logger.last_log(),
                )

                if self.verbose == 2 and val_steps:
                    progress_bar(epoch, vstep, self.logger.last_log())

        if self.verbose > 0:
            progress_bar.end_epoch(epoch, self.logger.last_log())

    def _run_training_step(self, batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.cfg.device is not None:
            batch = self._move_to_device(batch)
        return self._normalize_step_output(self.train_step(batch))

    def _run_validation_step(self, batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.cfg.device is not None:
            batch = self._move_to_device(batch)
        return self._normalize_step_output(self.validation_step(batch))

    def _move_to_device(self, batch: Any) -> Any:
        """
        Move batch tensors to the configured device.

        Override in subclasses to handle custom batch formats. The default
        implementation is a no-op; concrete trainers (e.g. Trainer) override
        this to handle (X, y) tuples.
        """
        return batch

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch):
        raise NotImplementedError

    @property
    def model(self) -> nn.Module:
        return self.cfg.model

    @model.setter
    def model(self, new_model: nn.Module) -> None:
        self.cfg.model = new_model

    @property
    def loss_fn(self):
        return self.cfg.loss_fn

    @loss_fn.setter
    def loss_fn(self, new_fn) -> None:
        self.cfg.loss_fn = new_fn

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.cfg.metrics

    @metrics.setter
    def metrics(self, new_m: Dict[str, Any]) -> None:
        self.cfg.metrics = new_m

    @property
    def optimizer(self) -> Optimizer:
        return self.cfg.optimizer

    @optimizer.setter
    def optimizer(self, new_opt: Optimizer) -> None:
        self.cfg.optimizer = new_opt

    @property
    def verbose(self) -> int:
        return self.cfg.verbose

    @verbose.setter
    def verbose(self, v: int) -> None:
        self.cfg.verbose = v

    def _on_train_epoch_start(self, epoch: int) -> None:
        """Called at the start of each training epoch before any batch is processed.

        Override in subclasses to reset epoch-local state or perform per-epoch
        setup (e.g. gradient accumulation counters, distributed sampler seeding).
        """

    def _compute_metrics(self, outputs: dict, prefix: str) -> dict:
        if not self.metrics:
            return {}

        if not isinstance(outputs, dict) or ("y_true" not in outputs) or ("y_pred" not in outputs):
            if self.cfg.strict:
                raise KeyError(
                    "Metrics are configured but `info` did not provide both 'y_true' and 'y_pred'. "
                    "Either return them in info, set strict=False, or remove metrics."
                )
            return {}

        y_true = outputs["y_true"]
        y_pred = outputs["y_pred"]

        results: Dict[str, Any] = {}
        with torch.inference_mode():
            for name, fn in self.metrics.items():
                val = fn(y_true=y_true, y_pred=y_pred)

                if hasattr(val, "detach"):
                    val = val.detach()
                if hasattr(val, "device"):
                    val = val.cpu()
                if hasattr(val, "numel") and val.numel() == 1:
                    val = float(val.item())

                results[f"{prefix}{name}"] = val

        return results

    def _normalize_step_output(self, result) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            metrics, info = result
            if metrics is None:
                metrics = {}
            if info is None:
                info = {}
            if not isinstance(metrics, dict) or not isinstance(info, dict):
                raise ValueError(
                    "If returning a tuple, it must be (metrics_dict, info_dict) where both are dicts."
                )
            return metrics, info
        if isinstance(result, dict):
            return result, {}
        return {"loss": result}, {}
