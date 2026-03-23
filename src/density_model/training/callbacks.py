"""
Training Callbacks
------------------
Callback base classes and common training callbacks for early stopping and model
checkpointing.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "EarlyStoppingConfig",
    "ModelCheckpoint",
    "ModelCheckpointConfig",
]


class Callback:
    """
    Base class for training callbacks.

    Parameters
    ----------
    None
        Subclasses define hook behavior.
    """

    trainer: Any

    def set_trainer(self, trainer: Any) -> None:
        """
        Attach a trainer instance to the callback.

        Parameters
        ----------
        trainer : Any
            Trainer instance invoking the callback.

        Returns
        -------
        None
            This method stores the provided trainer reference.
        """

        self.trainer = trainer

    def on_fit_start(self) -> None:
        """Run once before the training loop starts."""

    def on_fit_end(self) -> None:
        """Run once after the training loop finishes."""

    def on_epoch_start(self, epoch: int) -> None:
        """
        Run at the start of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        """

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """
        Run at the end of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict of str to Any
            Aggregated metric values from the trainer.
        """

    def on_train_step_start(self, step: int, batch: Any) -> None:
        """
        Run before a training step.

        Parameters
        ----------
        step : int
            One-based step index.
        batch : Any
            Batch about to be processed.
        """

    def on_train_step_end(self, step: int, batch: Any, outputs: dict[str, Any], logs: dict[str, Any]) -> None:
        """
        Run after a training step.

        Parameters
        ----------
        step : int
            One-based step index.
        batch : Any
            Batch that was processed.
        outputs : dict of str to Any
            Step outputs including metrics and auxiliary information.
        logs : dict of str to Any
            Aggregated trainer logs.
        """

    def on_validation_step_start(self, vstep: int, batch: Any) -> None:
        """
        Run before a validation step.

        Parameters
        ----------
        vstep : int
            One-based validation step index.
        batch : Any
            Batch about to be processed.
        """

    def on_validation_step_end(
        self,
        vstep: int,
        batch: Any,
        outputs: dict[str, Any],
        logs: dict[str, Any],
    ) -> None:
        """
        Run after a validation step.

        Parameters
        ----------
        vstep : int
            One-based validation step index.
        batch : Any
            Batch that was processed.
        outputs : dict of str to Any
            Step outputs including metrics and auxiliary information.
        logs : dict of str to Any
            Aggregated trainer logs.
        """

    def on_exception(self, exception: BaseException) -> None:
        """
        Run when an exception escapes the training loop.

        Parameters
        ----------
        exception : BaseException
            Exception raised during fitting.
        """


class CallbackList:
    """
    Dispatch callback lifecycle hooks to a sequence of callbacks.

    Parameters
    ----------
    callbacks : list of Callback, default=None
        Callback instances invoked in the order provided.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])
        self.trainer: Any | None = None

    def set_trainer(self, trainer: Any) -> None:
        """
        Attach the trainer to all callbacks.

        Parameters
        ----------
        trainer : Any
            Trainer instance invoking the callback list.

        Returns
        -------
        None
            This method attaches the trainer to all callbacks.
        """

        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def append(self, callback: Callback) -> None:
        """
        Append a callback to the list.

        Parameters
        ----------
        callback : Callback
            Callback to append.

        Returns
        -------
        None
            This method adds the callback to the sequence.
        """

        self.callbacks.append(callback)
        if self.trainer is not None:
            callback.set_trainer(self.trainer)

    def _call(self, name: str, *args: Any, **kwargs: Any) -> None:
        """
        Invoke a named hook on all callbacks.

        Parameters
        ----------
        name : str
            Hook method name.
        *args : Any
            Positional hook arguments.
        **kwargs : Any
            Keyword hook arguments.

        Returns
        -------
        None
            This method forwards hook calls in order.
        """

        for callback in self.callbacks:
            getattr(callback, name)(*args, **kwargs)

    def on_fit_start(self) -> None:
        """Dispatch ``on_fit_start`` to all callbacks."""

        self._call("on_fit_start")

    def on_fit_end(self) -> None:
        """Dispatch ``on_fit_end`` to all callbacks."""

        self._call("on_fit_end")

    def on_epoch_start(self, epoch: int) -> None:
        """
        Dispatch ``on_epoch_start`` to all callbacks.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        """

        self._call("on_epoch_start", epoch)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """
        Dispatch ``on_epoch_end`` to all callbacks.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict of str to Any
            Aggregated trainer logs.
        """

        self._call("on_epoch_end", epoch, logs)

    def on_train_step_start(self, step: int, batch: Any) -> None:
        """
        Dispatch ``on_train_step_start`` to all callbacks.

        Parameters
        ----------
        step : int
            One-based step index.
        batch : Any
            Batch about to be processed.
        """

        self._call("on_train_step_start", step, batch)

    def on_train_step_end(self, step: int, batch: Any, outputs: dict[str, Any], logs: dict[str, Any]) -> None:
        """
        Dispatch ``on_train_step_end`` to all callbacks.

        Parameters
        ----------
        step : int
            One-based step index.
        batch : Any
            Processed batch.
        outputs : dict of str to Any
            Step outputs.
        logs : dict of str to Any
            Aggregated trainer logs.
        """

        self._call("on_train_step_end", step, batch, outputs, logs)

    def on_validation_step_start(self, vstep: int, batch: Any) -> None:
        """
        Dispatch ``on_validation_step_start`` to all callbacks.

        Parameters
        ----------
        vstep : int
            One-based validation step index.
        batch : Any
            Batch about to be processed.
        """

        self._call("on_validation_step_start", vstep, batch)

    def on_validation_step_end(
        self,
        vstep: int,
        batch: Any,
        outputs: dict[str, Any],
        logs: dict[str, Any],
    ) -> None:
        """
        Dispatch ``on_validation_step_end`` to all callbacks.

        Parameters
        ----------
        vstep : int
            One-based validation step index.
        batch : Any
            Processed batch.
        outputs : dict of str to Any
            Step outputs.
        logs : dict of str to Any
            Aggregated trainer logs.
        """

        self._call("on_validation_step_end", vstep, batch, outputs, logs)

    def on_exception(self, exception: BaseException) -> None:
        """
        Dispatch ``on_exception`` to all callbacks.

        Parameters
        ----------
        exception : BaseException
            Exception raised during fitting.
        """

        self._call("on_exception", exception)


class EarlyStoppingConfig(BaseModel):
    """
    Pydantic configuration for early stopping behavior.

    Parameters
    ----------
    monitor : str, default="val_loss"
        Metric name to monitor.
    mode : {"min", "max"}, default="min"
        Optimization direction for the monitored metric.
    patience : int, default=10
        Number of epochs without improvement before stopping.
    min_delta : float, default=0.0
        Minimum improvement threshold.
    warmup : int, default=0
        Number of initial epochs ignored by early stopping.
    restore_best_weights : bool, default=True
        Whether to restore the best model state before stopping.
    verbose : bool, default=False
        Whether to print status messages.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    monitor: str = "val_loss"
    mode: str = Field(default="min", pattern="^(min|max)$")
    patience: int = Field(default=10, ge=0)
    min_delta: float = 0.0
    warmup: int = Field(default=0, ge=0)
    restore_best_weights: bool = True
    verbose: bool = False


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    config : EarlyStoppingConfig
        Validated callback configuration.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.config = config
        self.best: float | None = None
        self.wait = 0
        self.stopped_epoch: int | None = None
        self.best_weights: dict[str, Any] | None = None

    def _log(self, message: str) -> None:
        """
        Print a callback status message when verbose logging is enabled.

        Parameters
        ----------
        message : str
            User-facing callback status message.

        Returns
        -------
        None
            This method writes to standard output when enabled.
        """

        if self.config.verbose:
            print(message, flush=True)

    def _is_better(self, current: float, best: float) -> bool:
        """
        Determine whether the current metric improved over the best value.

        Parameters
        ----------
        current : float
            Current monitored metric value.
        best : float
            Best value observed so far.

        Returns
        -------
        bool
            Whether the metric improved according to the configured mode.
        """

        if self.config.mode == "min":
            return current < best - self.config.min_delta
        return current > best + self.config.min_delta

    def on_fit_start(self) -> None:
        """Reset internal state before fitting begins."""

        self.best = None
        self.wait = 0
        self.stopped_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """
        Evaluate early stopping criteria at the end of an epoch.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict of str to Any
            Aggregated trainer logs.

        Raises
        ------
        KeyError
            If the monitored metric is not present in the logs.
        """

        if self.config.monitor not in logs:
            raise KeyError(f"Monitored metric {self.config.monitor!r} not found in logs.")

        current = float(logs[self.config.monitor])
        if epoch < self.config.warmup:
            return
        if self.best is None:
            self.best = current
            if self.config.restore_best_weights:
                self.best_weights = deepcopy(self.trainer.model.state_dict())
            self._log(
                f"[EarlyStopping] epoch={epoch + 1} initialized best {self.config.monitor}={current:.6f}"
            )
            return
        if self._is_better(current=current, best=self.best):
            self.best = current
            self.wait = 0
            if self.config.restore_best_weights:
                self.best_weights = deepcopy(self.trainer.model.state_dict())
            self._log(
                f"[EarlyStopping] epoch={epoch + 1} improved {self.config.monitor} to {current:.6f}"
            )
            return

        self.wait += 1
        self._log(
            f"[EarlyStopping] epoch={epoch + 1} no improvement in {self.config.monitor} "
            f"(current={current:.6f}, best={self.best:.6f}, wait={self.wait}/{self.config.patience})"
        )
        if self.wait >= self.config.patience:
            self.stopped_epoch = epoch
            self.trainer.stop_training = True
            if self.config.restore_best_weights and self.best_weights is not None:
                self.trainer.model.load_state_dict(self.best_weights)
                self._log(
                    f"[EarlyStopping] stopping at epoch={epoch + 1} and restoring best weights "
                    f"from {self.config.monitor}={self.best:.6f}"
                )
            else:
                self._log(f"[EarlyStopping] stopping at epoch={epoch + 1}")


class ModelCheckpointConfig(BaseModel):
    """
    Pydantic configuration for model checkpointing.

    Parameters
    ----------
    filepath : str
        Path or directory used to save checkpoints.
    monitor : str, default="val_loss"
        Metric name to monitor.
    mode : {"min", "max"}, default="min"
        Optimization direction for the monitored metric.
    min_delta : float, default=0.0
        Minimum improvement threshold.
    warmup : int, default=0
        Number of initial epochs ignored by checkpointing.
    save_optimizer : bool, default=True
        Whether to save optimizer state.
    overwrite : bool, default=True
        Whether to overwrite the previous best checkpoint.
    verbose : bool, default=True
        Whether to print checkpoint messages.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    filepath: str
    monitor: str = "val_loss"
    mode: str = Field(default="min", pattern="^(min|max)$")
    min_delta: float = 0.0
    warmup: int = Field(default=0, ge=0)
    save_optimizer: bool = True
    overwrite: bool = True
    verbose: bool = True


class ModelCheckpoint(Callback):
    """
    Save model checkpoints when a monitored metric improves.

    Parameters
    ----------
    config : ModelCheckpointConfig
        Validated checkpoint configuration.
    """

    def __init__(self, config: ModelCheckpointConfig) -> None:
        self.config = config
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.last_saved_path: str | None = None

    def _log(self, message: str) -> None:
        """
        Print a checkpoint status message when verbose logging is enabled.

        Parameters
        ----------
        message : str
            User-facing callback status message.

        Returns
        -------
        None
            This method writes to standard output when enabled.
        """

        if self.config.verbose:
            print(message, flush=True)

    def on_fit_start(self) -> None:
        """Prepare checkpoint directory state at fit start."""

        resolved = Path(self._resolve_path())
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.best = None
        self.best_epoch = None
        self.last_saved_path = None

    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """
        Save a checkpoint if the monitored metric improved.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict of str to Any
            Aggregated trainer logs.

        Raises
        ------
        KeyError
            If the monitored metric is not present in the logs.
        """

        if self.config.monitor not in logs:
            raise KeyError(f"Monitored metric {self.config.monitor!r} not found in logs.")

        current = float(logs[self.config.monitor])
        if epoch < self.config.warmup:
            return

        if self.best is None or self._is_improved(current=current, best=self.best):
            self.best = current
            self.best_epoch = epoch
            path = self._resolve_path(epoch=None if self.config.overwrite else epoch)
            self._save_checkpoint(path=path, epoch=epoch)
            if self.config.overwrite and self.last_saved_path and self.last_saved_path != path:
                try:
                    os.remove(self.last_saved_path)
                except FileNotFoundError:
                    pass
            self.last_saved_path = path
            self._log(
                f"[ModelCheckpoint] epoch={epoch + 1} saved checkpoint to {path} "
                f"with {self.config.monitor}={current:.6f}"
            )

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        """
        Persist a model checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint output path.
        epoch : int
            Zero-based epoch index associated with the checkpoint.

        Returns
        -------
        None
            This method writes the checkpoint to disk.
        """

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.trainer.model.state_dict(),
            "best_metric": self.best,
        }
        if self.config.save_optimizer:
            checkpoint["optimizer_state_dict"] = self.trainer.optimizer.state_dict()
        torch.save(checkpoint, path)

    def _resolve_path(self, epoch: int | None = None) -> str:
        """
        Resolve the checkpoint path for the current save event.

        Parameters
        ----------
        epoch : int or None, default=None
            Epoch index used in non-overwrite mode.

        Returns
        -------
        str
            Resolved checkpoint path.
        """

        filepath = Path(self.config.filepath)
        if filepath.suffix:
            return str(filepath)
        if self.config.overwrite:
            return str(filepath / "best_model.pt")
        if epoch is None:
            raise ValueError("An epoch value is required when overwrite=False.")
        return str(filepath / f"checkpoint_epoch_{epoch}.pt")

    def _is_improved(self, current: float, best: float) -> bool:
        """
        Determine whether the current metric improved over the best value.

        Parameters
        ----------
        current : float
            Current monitored metric value.
        best : float
            Best value observed so far.

        Returns
        -------
        bool
            Whether the metric improved according to the configured mode.
        """

        if self.config.mode == "min":
            return current < best - self.config.min_delta
        return current > best + self.config.min_delta
