"""
Panel Tuning Pruning Callback
-----------------------------
Optuna pruning callback that reports the current monitored metric to the
in-flight trial at the end of every epoch and raises
``optuna.TrialPruned`` when Optuna decides the trial should stop early.

Classes
-------
PruningCallback
    Callback attached to the per-fold panel trainer during tuning.
"""

from __future__ import annotations

from typing import Any, Dict

from optuna import Trial, TrialPruned

from density_model.shared.training.callbacks import Callback

__all__ = ["PruningCallback"]


class PruningCallback(Callback):
    """
    Report a monitored metric to Optuna at the end of every epoch.

    Parameters
    ----------
    trial : optuna.Trial
        In-flight Optuna trial.
    monitor : str, default="val_loss"
        Metric name to read from ``trainer.logger.last_log()``.
    fold_step : int, default=0
        Outer fold step used so per-fold reports share a coherent timeline.
    """

    def __init__(self, *, trial: Trial, monitor: str = "val_loss", fold_step: int = 0) -> None:
        self.trial = trial
        self.monitor = monitor
        self.fold_step = fold_step

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=self.fold_step * 10_000 + epoch)
        if self.trial.should_prune():
            if hasattr(self, "trainer"):
                self.trainer.stop_training = True
            raise TrialPruned()
