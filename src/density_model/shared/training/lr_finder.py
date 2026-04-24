"""
Learning Rate Finder
--------------------
Implements the learning rate range test (Smith 2015) for identifying a
good initial learning rate before training.

Classes
-------
LRFinderResult
    Immutable result of a learning rate range test: parallel lists of
    learning rates and smoothed losses, with suggestion and plot helpers.
LRFinder
    Runs the range test over a trainer and dataloader, then restores all
    state so training can continue unaffected.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..training.base import BaseTrainer

__all__ = ["LRFinder", "LRFinderResult"]


@dataclass
class LRFinderResult:
    """
    Result of a learning rate range test.

    Attributes
    ----------
    lrs : list of float
        Learning rates tested, in ascending order.
    losses : list of float
        EMA-smoothed, bias-corrected loss recorded at each learning rate.

    Notes
    -----
    The lists are truncated at the point where loss diverged (if it did),
    so ``len(lrs)`` may be less than ``num_steps``.
    """

    lrs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    def suggest(self, method: str = "valley") -> float:
        """
        Suggest a learning rate from the range test results.

        Parameters
        ----------
        method : {"valley", "min"}, default "valley"
            - ``"valley"``: returns the LR at the point of steepest loss
              descent (most negative gradient). Robust and generally
              recommended.
            - ``"min"``: returns the LR one step before the loss minimum.
              Conservative; useful when the valley is ambiguous.

        Returns
        -------
        float
            Suggested learning rate.
        """
        if len(self.losses) < 2:
            raise RuntimeError("Too few data points to suggest a learning rate.")

        losses = np.array(self.losses)

        if method == "valley":
            grads = np.gradient(losses)
            idx = int(np.argmin(grads))
        elif method == "min":
            idx = max(0, int(np.argmin(losses)) - 1)
        else:
            raise ValueError(f"method must be 'valley' or 'min'; got {method!r}")

        return float(self.lrs[idx])

    def plot(self) -> None:
        """
        Plot loss vs learning rate (log scale).

        Requires matplotlib. Install with: ``pip install matplotlib``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for LRFinderResult.plot(). "
                "Install it with: pip install matplotlib"
            ) from exc

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogx(self.lrs, self.losses)
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Loss (smoothed)")
        ax.set_title("LR Finder — Range Test")
        ax.grid(True, which="both", alpha=0.3)

        try:
            suggested = self.suggest("valley")
            ax.axvline(x=suggested, color="red", linestyle="--", label=f"suggested: {suggested:.2e}")
            ax.legend()
        except RuntimeError:
            pass

        plt.tight_layout()
        plt.show()


class LRFinder:
    """
    Learning rate range test (Smith 2015).

    Sweeps the learning rate exponentially from ``start_lr`` to ``end_lr``
    over ``num_steps`` batches, recording the smoothed loss at each step.
    All model weights, optimizer state, and scaler state are restored after
    the run so training can continue from exactly where it was.

    Parameters
    ----------
    trainer : BaseTrainer
        Trainer whose model, optimizer, and loss function are used.

    Examples
    --------
    ::

        finder = LRFinder(trainer)
        result = finder.run(train_dataloader, start_lr=1e-7, end_lr=1.0)
        lr = result.suggest()
        print(f"Suggested LR: {lr:.2e}")
        result.plot()
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        self.trainer = trainer

    def run(
        self,
        train_dataloader,
        *,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_steps: int = 100,
        ema_decay: float = 0.98,
        diverge_threshold: float = 5.0,
    ) -> LRFinderResult:
        """
        Run the LR range test.

        Parameters
        ----------
        train_dataloader : DataLoader
            Source of training batches. Cycled automatically if exhausted
            before ``num_steps`` are completed.
        start_lr : float, default 1e-7
            Lower bound of the LR sweep.
        end_lr : float, default 10.0
            Upper bound of the LR sweep.
        num_steps : int, default 100
            Number of LR increments to test.
        ema_decay : float, default 0.98
            EMA decay for loss smoothing. Higher values produce smoother
            curves at the cost of responsiveness.
        diverge_threshold : float, default 5.0
            Stop early when the smoothed loss exceeds
            ``best_loss * diverge_threshold``.

        Returns
        -------
        LRFinderResult
        """
        trainer = self.trainer

        # Snapshot all state that will be mutated.
        model_state = deepcopy(trainer.model.state_dict())
        opt_state = deepcopy(trainer.optimizer.state_dict())
        scaler_state = deepcopy(trainer._scaler.state_dict()) if hasattr(trainer, "_scaler") else None

        # Temporarily disable gradient accumulation so each step is complete.
        orig_accum = trainer.cfg.grad_accumulation_steps
        trainer.cfg.grad_accumulation_steps = 1
        if hasattr(trainer, "_accum_count"):
            trainer._accum_count = 0

        lr_schedule = np.exp(
            np.linspace(np.log(start_lr), np.log(end_lr), num_steps)
        ).tolist()

        lrs: List[float] = []
        losses: List[float] = []
        smoothed_loss = 0.0
        best_loss = float("inf")

        loader_iter = iter(train_dataloader)
        trainer.model.train()

        try:
            for step_idx, lr in enumerate(lr_schedule):
                for group in trainer.optimizer.param_groups:
                    group["lr"] = lr

                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(train_dataloader)
                    batch = next(loader_iter)

                step_metrics, _ = trainer._run_training_step(batch)
                raw_loss = float(
                    step_metrics["loss"].item()
                    if hasattr(step_metrics["loss"], "item")
                    else step_metrics["loss"]
                )

                smoothed_loss = ema_decay * smoothed_loss + (1.0 - ema_decay) * raw_loss
                bias_corrected = smoothed_loss / (1.0 - ema_decay ** (step_idx + 1))

                lrs.append(lr)
                losses.append(bias_corrected)

                if bias_corrected < best_loss:
                    best_loss = bias_corrected

                if step_idx > 0 and bias_corrected > diverge_threshold * best_loss:
                    break

        finally:
            trainer.model.load_state_dict(model_state)
            trainer.optimizer.load_state_dict(opt_state)
            if scaler_state is not None:
                trainer._scaler.load_state_dict(scaler_state)
            trainer.cfg.grad_accumulation_steps = orig_accum
            if hasattr(trainer, "_accum_count"):
                trainer._accum_count = 0

        return LRFinderResult(lrs=lrs, losses=losses)
