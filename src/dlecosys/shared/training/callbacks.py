"""
Training Callbacks
-------------------
Callback base classes and concrete callback implementations for the
dlecosys training infrastructure.

Classes
-------
Callback
    Base class for training lifecycle hooks. Subclass to implement any
    combination of hooks; all are optional no-ops by default.
CallbackList
    Container that forwards lifecycle events to a list of Callback instances.
    Swallows exceptions by default to avoid interrupting training.
EarlyStopping
    Stops training when a monitored metric has stopped improving.
ModelCheckpoint
    Saves model weights whenever the monitored metric improves.
LRSchedulerCallback
    Steps a PyTorch learning rate scheduler at the end of each epoch.
GradNormCallback
    Logs the total gradient L2 norm at the end of each training step.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence

import torch

__all__ = [
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "LRSchedulerCallback",
    "GradNormCallback",
    "TensorBoardCallback",
]


class Callback:
    """
    Base class for training callbacks.

    Subclass to implement any hook(s) you need. All hooks are optional.
    Hooks receive simple, explicit arguments so they're easy to unit test.

    Notes
    -----
    - `trainer` is attached via `set_trainer` once, before training starts.
    - `logs` is a plain dict of running metrics (your MetricsTracker.last_log()).
    - Return values are ignored; modify trainer state or internal callback state.
    """

    def set_trainer(self, trainer: Any) -> None:
        """Attach the trainer instance so callbacks can access shared state."""
        self.trainer = trainer

    def on_fit_start(self) -> None:
        """Called once before the first epoch starts."""
        pass

    def on_fit_end(self) -> None:
        """Called once after the last epoch ends (or early stop)."""
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of an epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of an epoch with the latest aggregated logs."""
        pass

    def on_train_step_start(self, step: int, batch: Any) -> None:
        """Called right before a training step is executed."""
        pass

    def on_train_step_end(
        self,
        step: int,
        batch: Any,
        outputs: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> None:
        """Called right after a training step finishes."""
        pass

    def on_validation_step_start(self, vstep: int, batch: Any) -> None:
        """Called right before a validation step is executed."""
        pass

    def on_validation_step_end(
        self,
        vstep: int,
        batch: Any,
        outputs: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> None:
        """Called right after a validation step finishes."""
        pass

    def on_exception(self, exception: BaseException) -> None:
        """Called if an exception bubbles up during fit."""
        pass


class CallbackList:
    """
    Container that forwards hook calls to a list of callbacks.

    Safe-by-default: exceptions inside callbacks are swallowed to avoid
    interrupting training. Set `raise_errors=True` if you prefer propagation.
    """

    def __init__(
        self,
        callbacks: Optional[Sequence[Callback]] = None,
        *,
        raise_errors: bool = False,
    ) -> None:
        self.callbacks: List[Callback] = list(callbacks or [])
        self.raise_errors = raise_errors
        self._trainer: Any = None
        self._trainer_attached = False

    def set_trainer(self, trainer: Any) -> None:
        self._trainer = trainer
        for cb in self.callbacks:
            try:
                cb.set_trainer(trainer)
            except Exception:
                if self.raise_errors:
                    raise
        self._trainer_attached = True

    def append(self, cb: Callback) -> None:
        self.callbacks.append(cb)
        if self._trainer_attached:
            try:
                cb.set_trainer(self._trainer)
            except Exception:
                if self.raise_errors:
                    raise

    def _call(self, name: str, *args, **kwargs) -> None:
        for cb in self.callbacks:
            hook = getattr(cb, name, None)
            if hook is None:
                continue
            try:
                hook(*args, **kwargs)
            except Exception:
                if self.raise_errors:
                    raise

    def on_fit_start(self) -> None:
        self._call("on_fit_start")

    def on_fit_end(self) -> None:
        self._call("on_fit_end")

    def on_epoch_start(self, epoch: int) -> None:
        self._call("on_epoch_start", epoch)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        self._call("on_epoch_end", epoch, logs)

    def on_train_step_start(self, step: int, batch: Any) -> None:
        self._call("on_train_step_start", step, batch)

    def on_train_step_end(
        self,
        step: int,
        batch: Any,
        outputs: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> None:
        self._call("on_train_step_end", step, batch, outputs, logs)

    def on_validation_step_start(self, vstep: int, batch: Any) -> None:
        self._call("on_validation_step_start", vstep, batch)

    def on_validation_step_end(
        self,
        vstep: int,
        batch: Any,
        outputs: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> None:
        self._call("on_validation_step_end", vstep, batch, outputs, logs)

    def on_exception(self, exception: BaseException) -> None:
        self._call("on_exception", exception)


class EarlyStopping(Callback):
    """
    Early stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    monitor : str
        Name of the metric in `logs` to monitor.
    mode : {"min", "max"}
        If "min", lower is better. If "max", higher is better.
    patience : int
        Number of *post-warmup* epochs with no improvement after which training stops.
    min_delta : float
        Minimum change to qualify as an improvement.
    warmup : int
        Number of initial epochs (0-based) to ignore completely for early stopping.
        Epochs with index < warmup:
            - do NOT set `best`
            - do NOT increment `wait`
    restore_best_weights : bool
        If True, restore model weights from the epoch with the best monitored value.
    verbose : bool
        If True, prints status messages.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        warmup: int = 0,
        restore_best_weights: bool = True,
        verbose: bool = False,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = float(min_delta)
        self.warmup = int(warmup)
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best: Optional[float] = None
        self.wait: int = 0
        self.stopped_epoch: Optional[int] = None
        self.best_weights: Optional[Dict[str, Any]] = None

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        elif self.mode == "max":
            return current > best + self.min_delta
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

    def on_fit_start(self) -> None:
        self.best = None
        self.wait = 0
        self.stopped_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor, None)
        if current is None:
            if self.verbose:
                print(f"[EarlyStopping] Metric '{self.monitor}' not found in logs.")
            return

        if epoch < self.warmup:
            if self.verbose:
                print(
                    f"[EarlyStopping] Warmup epoch {epoch} "
                    f"(warmup={self.warmup}) – ignoring metric {self.monitor}."
                )
            return

        if self.best is None:
            self.best = float(current)
            if self.restore_best_weights:
                self.best_weights = deepcopy(self.trainer.model.state_dict())
            if self.verbose:
                print(
                    f"[EarlyStopping] Starting monitoring '{self.monitor}' "
                    f"at epoch {epoch} with value {current:.6f}."
                )
            return

        if self._is_better(float(current), float(self.best)):
            self.best = float(current)
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = deepcopy(self.trainer.model.state_dict())
            if self.verbose:
                print(
                    f"[EarlyStopping] Epoch {epoch}: '{self.monitor}' improved "
                    f"to {current:.6f}."
                )
        else:
            self.wait += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] Epoch {epoch}: no improvement in '{self.monitor}'. "
                    f"wait={self.wait}/{self.patience}"
                )

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer.stop_training = True

                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose:
                        print(
                            f"[EarlyStopping] Restoring best weights "
                            f"from epoch with {self.monitor}={self.best:.6f}."
                        )
                    self.trainer.model.load_state_dict(self.best_weights)

                if self.verbose:
                    print(
                        f"[EarlyStopping] Stopping at epoch {epoch} "
                        f"(warmup={self.warmup}, patience={self.patience})."
                    )


class ModelCheckpoint(Callback):
    """
    Model checkpoint callback for saving model weights during training.

    Saves the model whenever the monitored metric improves (after warmup).
    Optionally overwrites the previous checkpoint, keeping only the latest best file.

    Parameters
    ----------
    filepath : str
        Path or directory where the checkpoint will be saved.
        If a directory is provided, 'best_model.pt' is used as the filename.
    monitor : str
        Metric to monitor for improvement (e.g., "val_loss").
    mode : {"min", "max"}
        Whether the monitored metric should decrease or increase to count
        as an improvement.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    warmup : int
        Number of initial epochs (0-based) to ignore completely for checkpointing.
        Epochs with index < warmup:
            - do NOT set `best`
            - do NOT save checkpoints
    save_optimizer : bool
        Whether to include optimizer state in the checkpoint.
    overwrite : bool
        If True, overwrites the previous checkpoint instead of saving multiple files.
    verbose : bool
        If True, prints save messages to stdout.
    """

    def __init__(
        self,
        *,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        min_delta: float = 0.0,
        warmup: int = 0,
        save_optimizer: bool = True,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")

        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)
        self.warmup = int(warmup)
        self.save_optimizer = save_optimizer
        self.overwrite = overwrite
        self.verbose = verbose

        self._best: Optional[float] = None
        self._best_epoch: Optional[int] = None
        self._last_saved_path: Optional[str] = None

    def on_fit_start(self) -> None:
        parent = os.path.dirname(self._resolve_path())
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._best = None
        self._best_epoch = None
        self._last_saved_path = None

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.monitor not in logs:
            if self.verbose and epoch == 0:
                print(f"[ModelCheckpoint] Warning: '{self.monitor}' not found in logs.")
            return

        current = logs[self.monitor]
        if hasattr(current, "item"):
            current = float(current.item())
        else:
            current = float(current)

        if epoch < self.warmup:
            if self.verbose:
                print(
                    f"[ModelCheckpoint] Warmup epoch {epoch} "
                    f"(warmup={self.warmup}) – skipping checkpoint."
                )
            return

        if self._best is None:
            self._best = current
            self._best_epoch = epoch
            path = self._resolve_path(epoch if not self.overwrite else None)
            self._save_checkpoint(path, epoch)
            if self.verbose:
                print(
                    f"[ModelCheckpoint] Starting monitoring '{self.monitor}' "
                    f"at epoch {epoch}, initial best={self._best:.6f} "
                    f"(checkpoint saved to {path})."
                )
            self._last_saved_path = path
            return

        if self._is_improved(current, self._best):
            self._best = current
            self._best_epoch = epoch
            path = self._resolve_path(epoch if not self.overwrite else None)
            self._save_checkpoint(path, epoch)

            if self.verbose:
                print(
                    f"[ModelCheckpoint] Epoch {epoch}: '{self.monitor}' improved "
                    f"to {self._best:.6f} – saving checkpoint to {path}."
                )

            if self.overwrite and self._last_saved_path and self._last_saved_path != path:
                try:
                    os.remove(self._last_saved_path)
                    if self.verbose:
                        print(
                            f"[ModelCheckpoint] Removed previous checkpoint: "
                            f"{self._last_saved_path}"
                        )
                except FileNotFoundError:
                    pass

            self._last_saved_path = path
        else:
            if self.verbose:
                print(
                    f"[ModelCheckpoint] Epoch {epoch}: '{self.monitor}' "
                    f"did not improve (current={current:.6f}, best={self._best:.6f})."
                )

    def _save_checkpoint(self, path: str, epoch: int) -> None:
        from dlecosys.shared.training.process_group import is_main_process
        if not is_main_process():
            return
        model = self.trainer.model
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "best_metric": self._best,
        }
        if self.save_optimizer:
            checkpoint["optimizer_state_dict"] = self.trainer.optimizer.state_dict()
        torch.save(checkpoint, path)

    def _resolve_path(self, epoch: Optional[int] = None) -> str:
        if os.path.isdir(self.filepath):
            if self.overwrite:
                return os.path.join(self.filepath, "best_model.pt")
            else:
                return os.path.join(self.filepath, f"checkpoint_epoch_{epoch}.pt")
        return self.filepath

    def _is_improved(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class LRSchedulerCallback(Callback):
    """
    Step a PyTorch learning rate scheduler at the end of each epoch.

    Supports both standard schedulers (e.g. StepLR, CosineAnnealingLR) and
    metric-aware schedulers (e.g. ReduceLROnPlateau). For the latter, pass
    ``monitor`` to specify which logged metric to forward to ``scheduler.step``.

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Any PyTorch learning rate scheduler.
    monitor : str, optional
        Name of the metric in ``logs`` to pass to schedulers that require a
        metric value (e.g. ``ReduceLROnPlateau``). If None, ``scheduler.step()``
        is called with no arguments.
    verbose : bool, default False
        If True, prints the updated learning rate(s) after each step.
    """

    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.monitor is not None:
            metric_val = logs.get(self.monitor)
            if metric_val is None:
                if self.verbose:
                    print(
                        f"[LRSchedulerCallback] Monitor '{self.monitor}' not found "
                        f"in logs at epoch {epoch}. Scheduler not stepped."
                    )
                return
            self.scheduler.step(metric_val)
        else:
            self.scheduler.step()

        if self.verbose:
            lrs = [f"{pg['lr']:.2e}" for pg in self.trainer.optimizer.param_groups]
            lr_str = lrs[0] if len(lrs) == 1 else f"[{', '.join(lrs)}]"
            print(f"[LRSchedulerCallback] Epoch {epoch}: lr={lr_str}")


class GradNormCallback(Callback):
    """
    Logs the total gradient L2 norm at the end of each training step.

    Useful for monitoring optimization health: a stable, bounded norm
    indicates healthy training; a growing or NaN norm signals instability.

    Parameters
    ----------
    log_key : str, default "grad_norm"
        Key under which the gradient norm is written to the trainer logger.

    Notes
    -----
    With gradient accumulation (``grad_accumulation_steps > 1``), the logged
    norm grows across each accumulation window and is largest — and most
    meaningful — at the accumulation boundary just after the optimizer step.
    With AMP enabled, norms at non-boundary steps are in the scaled gradient
    domain and should be interpreted as relative magnitudes only.
    """

    def __init__(self, log_key: str = "grad_norm") -> None:
        self.log_key = log_key

    def on_train_step_end(
        self,
        step: int,
        batch: Any,
        outputs: Dict[str, Any],
        logs: Dict[str, Any],
    ) -> None:
        norm_sq = sum(
            p.grad.detach().norm(2).item() ** 2
            for p in self.trainer.model.parameters()
            if p.grad is not None
        )
        self.trainer.logger.update_state(**{self.log_key: norm_sq ** 0.5})


class TensorBoardCallback(Callback):
    """
    Write every scalar in ``logs`` to a TensorBoard event file at each epoch end.

    Parameters
    ----------
    log_dir : str
        Directory where the SummaryWriter will write event files.
        Typically ``<run>/logs/tensorboard``.

    Notes
    -----
    - Uses ``torch.utils.tensorboard.SummaryWriter`` which ships with PyTorch.
      The ``tensorboard`` package is only needed to VIEW the logs
      (``tensorboard --logdir <log_dir>``), not to write them.
    - Only numeric scalars are forwarded; non-scalars in ``logs`` are skipped.
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._writer = None

    def on_fit_start(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoardCallback requires the 'tensorboard' package. "
                "Install it with: pip install tensorboard"
            ) from exc
        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self._writer is None:
            return
        for key, value in logs.items():
            try:
                self._writer.add_scalar(key, float(value), epoch)
            except (TypeError, ValueError):
                continue

    def on_fit_end(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
