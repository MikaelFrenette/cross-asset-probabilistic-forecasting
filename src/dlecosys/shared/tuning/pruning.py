"""
Pruning Callback
----------------
Bridge between the training callback system and Optuna's pruning API.

Classes
-------
PruningCallback
    Reports the monitored validation metric to Optuna at each epoch and
    raises TrialPruned if the trial is deemed unpromising.
"""

from __future__ import annotations

from typing import Any, Dict

import optuna

from dlecosys.shared.training.callbacks import Callback

__all__ = ["PruningCallback"]


class PruningCallback(Callback):
    """
    Report intermediate metrics to an Optuna trial and prune if needed.

    Parameters
    ----------
    trial : optuna.Trial
        The trial receiving intermediate values.
    monitor : str, default "val_loss"
        Key in the logs dict to forward to the trial at each epoch end.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val_loss") -> None:
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        value = logs.get(self.monitor)
        if value is None:
            return
        if hasattr(value, "item"):
            value = float(value.item())
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
