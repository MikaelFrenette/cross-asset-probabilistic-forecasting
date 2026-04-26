"""
Panel Tuning Pruning Callback
-----------------------------
Optuna pruning callback that reports the current monitored metric to the
in-flight trial at the end of every epoch. When Optuna decides the trial
should prune, the callback sets an internal ``should_prune`` flag and
flips ``trainer.stop_training`` so the fold's epoch loop exits cleanly.

dlecosys' :class:`CallbackList` is safe-by-default and swallows every
exception raised inside a callback, so raising ``optuna.TrialPruned``
directly from ``on_epoch_end`` never reaches the objective. Instead the
tuning objective must inspect the callback's ``should_prune`` flag after
``trainer.train()`` returns and raise ``TrialPruned`` itself — see
:meth:`PanelObjective._evaluate_fold`.

Classes
-------
PruningCallback
    Callback attached to the per-fold panel trainer during tuning.
"""

from __future__ import annotations

from typing import Any, Dict

from optuna import Trial

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

    Attributes
    ----------
    should_prune : bool
        Set to ``True`` once Optuna's ``should_prune`` returns true. The
        objective reads this after ``trainer.train()`` returns and raises
        :class:`optuna.TrialPruned` so Optuna records the correct state.
    """

    def __init__(self, *, trial: Trial, monitor: str = "val_loss", fold_step: int = 0) -> None:
        self.trial = trial
        self.monitor = monitor
        self.fold_step = fold_step
        self.should_prune = False

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=self.fold_step * 10_000 + epoch)
        if self.trial.should_prune():
            self.should_prune = True
            if hasattr(self, "trainer"):
                self.trainer.stop_training = True
