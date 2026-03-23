"""
Base Trainer
------------
Abstract training loop backbone for PyTorch models with callback dispatch,
metric tracking, and validation hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from density_model.config.trainer import BaseTrainerConfig
from density_model.training.callbacks import CallbackList
from density_model.training.utils import MetricsTracker, ProgressBar

__all__ = ["BaseTrainer"]


class BaseTrainer(ABC):
    """
    Abstract trainer coordinating training and validation loops.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments validated by ``BaseTrainerConfig``.

    Attributes
    ----------
    cfg : BaseTrainerConfig
        Validated trainer configuration.
    logger : MetricsTracker
        Metric tracker used for current logs and history.
    callbacks : CallbackList
        Callback dispatcher for trainer lifecycle hooks.
    stop_training : bool
        Flag used by callbacks to stop training early.
    """

    __slots__ = ("cfg", "logger", "callbacks", "stop_training", "__dict__")

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = BaseTrainerConfig(**kwargs)
        self.cfg.model.to(self.device)
        self.logger = MetricsTracker()
        self.logger.reset_state()
        self.callbacks = CallbackList(self.cfg.callbacks)
        self.callbacks.set_trainer(self)
        self.stop_training = False

    def train(
        self,
        *,
        train_dataloader: Any,
        epochs: int,
        val_dataloader: Any | None = None,
        initial_epoch: int = 0,
    ) -> None:
        """
        Execute the full training loop.

        Parameters
        ----------
        train_dataloader : Any
            Iterable of training batches.
        epochs : int
            Number of training epochs.
        val_dataloader : Any or None, default=None
            Optional iterable of validation batches.
        initial_epoch : int, default=0
            Zero-based epoch index from which training should resume.

        Returns
        -------
        None
            This method runs the trainer in place.
        """

        if initial_epoch < 0:
            raise ValueError("initial_epoch must be greater than or equal to zero.")
        if initial_epoch > epochs:
            raise ValueError("initial_epoch must be less than or equal to epochs.")

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
            for epoch in range(initial_epoch, epochs):
                if self.stop_training:
                    break

                self.callbacks.on_epoch_start(epoch)
                samples_seen = 0
                global_train_metrics: dict[str, float] = {}
                self.cfg.model.train()

                for step, batch in enumerate(train_dataloader, start=1):
                    self.callbacks.on_train_step_start(step, batch)
                    step_metrics, info = self._run_training_step(batch)
                    step_metrics = {
                        key: (float(value.item()) if hasattr(value, "item") else float(value))
                        for key, value in step_metrics.items()
                    }
                    batch_size = self._infer_batch_size(batch)
                    samples_seen += batch_size
                    train_metrics = self._compute_metrics(info, prefix="train_")
                    step_metrics.update(train_metrics)

                    for key, value in step_metrics.items():
                        global_train_metrics[key] = global_train_metrics.get(key, 0.0) + value * batch_size

                    logged_metrics = {
                        key: global_train_metrics[key] / samples_seen for key in global_train_metrics
                    }
                    self.logger.update_state(**logged_metrics)
                    self.callbacks.on_train_step_end(
                        step=step,
                        batch=batch,
                        outputs={"metrics": step_metrics, "info": info},
                        logs=self.logger.last_log(),
                    )

                    if self.verbose == 2 and steps_per_epoch:
                        progress_bar(epoch, step, self.logger.last_log())

                if val_dataloader is not None:
                    self.cfg.model.eval()
                    samples_seen = 0
                    global_val_metrics: dict[str, float] = {}
                    with torch.no_grad():
                        for validation_step, val_batch in enumerate(val_dataloader, start=1):
                            self.callbacks.on_validation_step_start(validation_step, val_batch)
                            val_step_metrics, info = self._run_validation_step(val_batch)
                            val_step_metrics = {
                                f"val_{key}": (float(value.item()) if hasattr(value, "item") else float(value))
                                for key, value in val_step_metrics.items()
                            }
                            batch_size = self._infer_batch_size(val_batch)
                            samples_seen += batch_size
                            val_metrics = self._compute_metrics(info, prefix="val_")
                            val_step_metrics.update(val_metrics)

                            for key, value in val_step_metrics.items():
                                global_val_metrics[key] = global_val_metrics.get(key, 0.0) + value * batch_size

                            val_logged_metrics = {
                                key: global_val_metrics[key] / samples_seen for key in global_val_metrics
                            }
                            self.logger.update_state(**val_logged_metrics)
                            self.callbacks.on_validation_step_end(
                                vstep=validation_step,
                                batch=val_batch,
                                outputs={"metrics": val_step_metrics, "info": info},
                                logs=self.logger.last_log(),
                            )

                            if self.verbose == 2 and steps_per_epoch:
                                progress_bar(epoch, validation_step, self.logger.last_log())

                if self.verbose > 0:
                    progress_bar.end_epoch(epoch, self.logger.last_log())

                self.logger.push(epoch=epoch, step=steps_per_epoch if steps_per_epoch else None)
                self.callbacks.on_epoch_end(epoch, self.logger.last_log())

                if self.stop_training:
                    break
        except BaseException as exception:
            self.callbacks.on_exception(exception)
            raise
        finally:
            self.callbacks.on_fit_end()

    def _run_training_step(self, batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Execute and normalize a training step result.

        Parameters
        ----------
        batch : Any
            Batch to pass to ``train_step``.

        Returns
        -------
        tuple of dict and dict
            Normalized metrics and auxiliary information dictionaries.
        """

        return self._normalize_step_output(self.train_step(batch))

    def _run_validation_step(self, batch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Execute and normalize a validation step result.

        Parameters
        ----------
        batch : Any
            Batch to pass to ``validation_step``.

        Returns
        -------
        tuple of dict and dict
            Normalized metrics and auxiliary information dictionaries.
        """

        return self._normalize_step_output(self.validation_step(batch))

    @abstractmethod
    def train_step(self, batch: Any) -> Any:
        """
        Execute one training step.

        Parameters
        ----------
        batch : Any
            Batch to process.
        """

    @abstractmethod
    def validation_step(self, batch: Any) -> Any:
        """
        Execute one validation step.

        Parameters
        ----------
        batch : Any
            Batch to process.
        """

    @property
    def model(self) -> nn.Module:
        """
        Return the configured model.

        Returns
        -------
        torch.nn.Module
            Configured model instance.
        """

        return self.cfg.model

    @model.setter
    def model(self, new_model: nn.Module) -> None:
        """
        Update the configured model with validation.

        Parameters
        ----------
        new_model : torch.nn.Module
            Replacement model.
        """

        self.cfg.model = new_model

    @property
    def loss_fn(self) -> Any:
        """
        Return the configured loss function.

        Returns
        -------
        Any
            Configured loss object.
        """

        return self.cfg.loss_fn

    @loss_fn.setter
    def loss_fn(self, new_loss_fn: Any) -> None:
        """
        Update the configured loss function with validation.

        Parameters
        ----------
        new_loss_fn : Any
            Replacement loss object.
        """

        self.cfg.loss_fn = new_loss_fn

    @property
    def metrics(self) -> dict[str, Any]:
        """
        Return the configured metrics.

        Returns
        -------
        dict of str to Any
            Metric mapping keyed by metric name.
        """

        return self.cfg.metrics

    @metrics.setter
    def metrics(self, new_metrics: dict[str, Any]) -> None:
        """
        Update the configured metric mapping with validation.

        Parameters
        ----------
        new_metrics : dict of str to Any
            Replacement metric mapping.
        """

        self.cfg.metrics = new_metrics

    @property
    def optimizer(self) -> Optimizer:
        """
        Return the configured optimizer.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer instance.
        """

        return self.cfg.optimizer

    @optimizer.setter
    def optimizer(self, new_optimizer: Optimizer) -> None:
        """
        Update the configured optimizer with validation.

        Parameters
        ----------
        new_optimizer : torch.optim.Optimizer
            Replacement optimizer.
        """

        self.cfg.optimizer = new_optimizer

    @property
    def device(self) -> torch.device:
        """
        Return the configured torch device.

        Returns
        -------
        torch.device
            Device used for model and batch placement.
        """

        return torch.device(self.cfg.device)

    @device.setter
    def device(self, value: str) -> None:
        """
        Update the configured device and move the model accordingly.

        Parameters
        ----------
        value : str
            Replacement torch device identifier.
        """

        self.cfg.device = value
        self.cfg.model.to(torch.device(self.cfg.device))

    @property
    def verbose(self) -> int:
        """
        Return the configured verbosity level.

        Returns
        -------
        int
            Verbosity level in ``{0, 1, 2}``.
        """

        return self.cfg.verbose

    @verbose.setter
    def verbose(self, value: int) -> None:
        """
        Update the configured verbosity level with validation.

        Parameters
        ----------
        value : int
            Replacement verbosity level.
        """

        self.cfg.verbose = value

    def _compute_metrics(self, outputs: dict[str, Any], prefix: str) -> dict[str, float]:
        """
        Compute configured metrics from step outputs.

        Parameters
        ----------
        outputs : dict of str to Any
            Auxiliary step outputs expected to include ``y_true`` and ``y_pred``.
        prefix : str
            Metric name prefix such as ``train_`` or ``val_``.

        Returns
        -------
        dict of str to float
            Computed scalar metrics.

        Raises
        ------
        KeyError
            If strict mode is enabled and required outputs are missing.
        """

        if not self.metrics:
            return {}
        if not isinstance(outputs, dict) or "y_true" not in outputs or "y_pred" not in outputs:
            if self.cfg.strict:
                raise KeyError(
                    "Metrics are configured but step info did not provide both 'y_true' and 'y_pred'."
                )
            return {}

        y_true = outputs["y_true"]
        y_pred = outputs["y_pred"]
        results: dict[str, float] = {}
        with torch.inference_mode():
            for name, metric_fn in self.metrics.items():
                value = metric_fn(y_true, y_pred)
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "numel") and value.numel() == 1:
                    value = float(value.item())
                results[f"{prefix}{name}"] = float(value)
        return results

    def _normalize_step_output(self, result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Normalize step outputs into ``(metrics, info)`` form.

        Parameters
        ----------
        result : Any
            Raw result returned by ``train_step`` or ``validation_step``.

        Returns
        -------
        tuple of dict and dict
            Normalized metrics and auxiliary information dictionaries.

        Raises
        ------
        ValueError
            If a tuple result does not contain two dictionaries.
        """

        if isinstance(result, tuple) and len(result) == 2:
            metrics, info = result
            if metrics is None:
                metrics = {}
            if info is None:
                info = {}
            if not isinstance(metrics, dict) or not isinstance(info, dict):
                raise ValueError(
                    "If returning a tuple, it must be (metrics_dict, info_dict) where both are dictionaries."
                )
            return metrics, info
        if isinstance(result, dict):
            return result, {}
        return {"loss": result}, {}

    def move_to_device(self, value: Any) -> Any:
        """
        Move tensors within a nested batch structure to the configured device.

        Parameters
        ----------
        value : Any
            Tensor or nested container to move.

        Returns
        -------
        Any
            Structure with tensor leaves moved to ``self.device``.
        """

        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self.move_to_device(nested) for key, nested in value.items()}
        if isinstance(value, tuple):
            return tuple(self.move_to_device(nested) for nested in value)
        if isinstance(value, list):
            return [self.move_to_device(nested) for nested in value]
        return value

    def _infer_batch_size(self, batch: Any) -> int:
        """
        Infer the number of samples represented by a batch.

        Parameters
        ----------
        batch : Any
            Batch object emitted by the dataloader.

        Returns
        -------
        int
            Inferred sample count used for metric weighting.
        """

        inferred = self._extract_batch_size(batch)
        return inferred if inferred is not None else 1

    def _extract_batch_size(self, value: Any) -> int | None:
        """
        Recursively extract batch size information from nested batch values.

        Parameters
        ----------
        value : Any
            Value to inspect.

        Returns
        -------
        int or None
            Leading batch dimension if it can be inferred, otherwise ``None``.
        """

        if torch.is_tensor(value):
            if value.ndim == 0:
                return 1
            return int(value.shape[0])
        if isinstance(value, dict):
            for nested in value.values():
                inferred = self._extract_batch_size(nested)
                if inferred is not None:
                    return inferred
            return None
        if isinstance(value, (tuple, list)):
            for nested in value:
                inferred = self._extract_batch_size(nested)
                if inferred is not None:
                    return inferred
            return None
        return None
