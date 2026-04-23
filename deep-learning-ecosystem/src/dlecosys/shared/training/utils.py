"""
Training Utilities
-------------------
Shared utilities for training loop instrumentation.

Classes
-------
MetricsTracker
    Accumulates running metrics during training and records periodic snapshots
    as an inspectable history.
ProgressBar
    Lightweight console progress bar for epoch/step-based training loops with
    ETA estimation and optional ANSI coloring.
LogRow
    Immutable snapshot of metrics recorded at a point in time.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import torch

__all__ = ["MetricsTracker", "ProgressBar", "LogRow"]


@dataclass(frozen=True)
class LogRow:
    """
    Immutable log snapshot recorded at a point in time (e.g., end of epoch).
    """
    step: Optional[int]
    epoch: Optional[int]
    metrics: Dict[str, float]


class MetricsTracker:
    """
    Metrics tracking utility for training loops.

    Design goals
    ------------
    - Robust: store only scalar floats (no tensors, no numpy scalars lingering).
    - Friendly for progress display: expose `last_log()` as a flat dict[str, float].
    - Friendly for analysis: record periodic snapshots in `history` as a DataFrame.
    - Minimal surface area: intended to be used by BaseTrainer, not as a full logging framework.

    Typical usage
    -------------
    - Call `update_state(...)` frequently (e.g., every step) to set the *current* metrics.
    - Call `push(epoch=..., step=...)` at desired cadence (e.g., end of epoch) to persist a snapshot row.

    Notes
    -----
    - `update_state` overwrites keys in the current log state.
    - `push` appends a row to history; it does not clear the current state.
    """

    def __init__(self) -> None:
        self._current: Dict[str, float] = {}
        self._rows: List[LogRow] = []


    def _to_float(self, value: Any, *, name: str) -> float:
        """
        Convert a metric value to a Python float.

        Accepts:
          - int / float
          - 0-d or 1-element torch.Tensor

        Raises:
          TypeError / ValueError for unsupported shapes or non-finite values.
        """
        if isinstance(value, bool):
            raise TypeError(f"Metric '{name}' must be numeric; got bool.")

        if isinstance(value, (int, float)):
            out = float(value)
        elif torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Metric '{name}' must be a scalar; got tensor with numel={value.numel()}.")
            out = float(value.detach().item())
        else:
            raise TypeError(f"Metric '{name}' must be int/float/torch scalar tensor; got {type(value)}.")

        if not math.isfinite(out):
            raise ValueError(f"Metric '{name}' must be finite; got {out!r}.")
        return out

    # -------------------------
    # Public API
    # -------------------------
    def reset_state(self) -> None:
        self._current.clear()
        self._rows.clear()

    def clear_current(self) -> None:
        """
        Clear only the current metrics (does not erase history).
        """
        self._current.clear()

    def update_state(self, values: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        """
        Update current metrics.

        Parameters
        ----------
        values:
            Optional mapping of metric_name -> metric_value.
        **kwargs:
            Additional metric_name -> metric_value entries.

        Notes
        -----
        - Later updates overwrite earlier keys in the current state.
        - Values are converted to floats and validated.
        """
        merged: Dict[str, Any] = {}
        if values:
            merged.update(dict(values))
        merged.update(kwargs)

        for k, v in merged.items():
            if not isinstance(k, str) or not k:
                raise TypeError("Metric names must be non-empty strings.")
            self._current[k] = self._to_float(v, name=k)

    def last_log(self) -> Dict[str, float]:
        """
        Return the latest current metrics snapshot (flat dict).
        """
        return dict(self._current)

    def push(self, *, epoch: Optional[int] = None, step: Optional[int] = None) -> None:
        """
        Persist the current metrics as a history row.

        Typical usage: call once per epoch (or at any logging cadence you prefer).
        """
        self._rows.append(LogRow(step=step, epoch=epoch, metrics=dict(self._current)))

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute min/max/avg over history rows for each metric.

        Returns empty dict if no history exists.
        """
        if not self._rows:
            return {}

        # Collect metric series across rows
        series: Dict[str, List[float]] = {}
        for row in self._rows:
            for k, v in row.metrics.items():
                series.setdefault(k, []).append(v)

        out: Dict[str, Dict[str, float]] = {}
        for k, vals in series.items():
            if vals:
                out[k] = {
                    "min": float(min(vals)),
                    "max": float(max(vals)),
                    "avg": float(sum(vals) / len(vals)),
                }
        return out

    @property
    def history(self) -> pd.DataFrame:
        """
        History as a DataFrame (one row per `push()`).

        Columns include:
          - epoch (if provided)
          - step (if provided)
          - metric keys

        Returns an empty DataFrame if no snapshots have been pushed yet.
        """
        if not self._rows:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for r in self._rows:
            d: Dict[str, Any] = {"epoch": r.epoch, "step": r.step}
            d.update(r.metrics)
            rows.append(d)
        return pd.DataFrame(rows)
            
class ProgressBar:
    """
    Lightweight console progress bar for epoch/step–based training loops.

    This class is designed to integrate with trainer-style loops where progress
    is reported at each step (or only at epoch boundaries, depending on
    verbosity). It provides:

      - A fixed-width progress bar updated in-place
      - Per-epoch ETA estimation based on smoothed step timing
      - Optional ANSI coloring when attached to a TTY
      - Graceful degradation when the number of steps per epoch is unknown

    Parameters
    ----------
    name : str
        Display name shown at the beginning of each progress line
        (e.g. "Training", "Validation").
    total_epochs : int
        Total number of epochs in the run.
    steps_per_epoch : int
        Number of steps in one epoch. If set to 0 or unknown, the bar will
        display `?/` for steps and disable ETA estimation.
    length : int, default=20
        Width of the progress bar in characters.
    fill : str, default="▄"
        Character used for the filled portion of the bar.
    eta_smoothing : float, default=0.2
        Exponential smoothing factor applied to the per-step time estimate.
        Lower values produce smoother but less reactive ETAs.

    Notes
    -----
    - The progress bar assumes:
        * `epoch` is a 0-based index
        * `step` is a 1-based index (as produced by `enumerate(dataloader, start=1)`)
    - ETA is computed *within each epoch* and resets at the end of an epoch.
    - When stdout is not a TTY (e.g., logs redirected to file), ANSI escape
      codes are disabled automatically.

    Methods
    -------
    __call__(epoch, step, logs)
        Update the progress bar for the current step.
    end_epoch(epoch, logs=None)
        Finalize and print a completed progress bar for the epoch on its own line.
    """
    def __init__(
        self,
        name: str,
        total_epochs: int,
        steps_per_epoch: int,
        length: int = 20,
        fill: str = "▄",
        eta_smoothing: float = 0.2,
    ):
        self.name = name
        self.length = length
        self.fill = fill
        self.total_epochs = int(total_epochs)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else 0

        self.eta_smoothing = float(eta_smoothing)
        self.start_time: Optional[float] = None
        self._last_logs: dict = {}

        self._ema_step_time: Optional[float] = None
        self._use_ansi = sys.stdout.isatty()

    def _format_eta(self, eta_seconds: int) -> str:
        eta_seconds = max(int(eta_seconds), 0)
        hours, remainder = divmod(eta_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"

    def _create_progress_bar(self, step: int, steps_per_epoch: int) -> str:
        if steps_per_epoch <= 0:
            return "-" * self.length
        step = max(0, min(int(step), int(steps_per_epoch)))
        filled_length = int(self.length * step // steps_per_epoch)
        filled = self.fill * filled_length
        empty = "-" * (self.length - filled_length)
        if self._use_ansi:
            return f"\033[92m{filled}\033[0m{empty}"
        return f"{filled}{empty}"

    def _reset_epoch_timer(self) -> None:
        self.start_time = time.time()
        self._ema_step_time = None

    def __call__(self, epoch: int, step: int, logs: dict):
        """
        Expected indexing:
          - epoch: 0-based epoch index
          - step:  1..steps_per_epoch (as in enumerate(dataloader, start=1))
        """
        if self.start_time is None:
            self._reset_epoch_timer()

        self._last_logs = logs or {}

        if self.steps_per_epoch <= 0:
            eta_formatted = "--:--"
            tps_str = "--"
            bar = self._create_progress_bar(0, 0)
            step_display = int(step)
            steps_display = "?"
        else:
            step_idx = max(1, min(int(step), int(self.steps_per_epoch)))
            elapsed = time.time() - float(self.start_time)

            avg_time_per_step = elapsed / max(step_idx, 1)

            if self._ema_step_time is None:
                self._ema_step_time = avg_time_per_step
            else:
                a = self.eta_smoothing
                self._ema_step_time = a * avg_time_per_step + (1.0 - a) * self._ema_step_time

            steps_left = self.steps_per_epoch - step_idx
            eta_seconds = int(max(steps_left, 0) * float(self._ema_step_time))
            eta_formatted = self._format_eta(eta_seconds)

            # time-per-step string (ms or s)
            tps = float(self._ema_step_time)
            tps_str = f"{tps * 1000.0:.0f}ms/step" if tps < 1.0 else f"{tps:.2f}s/step"

            bar = self._create_progress_bar(step_idx, self.steps_per_epoch)
            step_display = step_idx
            steps_display = self.steps_per_epoch

        parts = [
            "\r",
            f"{self.name} - Epoch: {epoch + 1}/{self.total_epochs}",
            f"|{bar}| Step: {step_display}/{steps_display}",
            f"- ETA: {eta_formatted}",
            f"- {tps_str}",
        ]

        for name, val in self._last_logs.items():
            parts.append(f"- {name}: {val:.4f}")

        sys.stdout.write(" ".join(parts))
        sys.stdout.flush()

    def reset(self, steps_per_epoch: int, name: Optional[str] = None) -> None:
        """
        Reset the bar for a new phase within the same epoch.

        Typically called when switching from the training phase to validation:
        updates the step count and display name, then clears the ETA timer so
        it restarts on the first call of the new phase.

        Parameters
        ----------
        steps_per_epoch : int
            Number of steps in the new phase. Pass 0 if unknown.
        name : str, optional
            Display name for the new phase (e.g. "Validation").
        """
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else 0
        if name is not None:
            self.name = name
        self.start_time = None
        self._ema_step_time = None

    def end_epoch(self, epoch: int, logs: dict | None = None):
        """
        Freeze a full bar for the completed phase on its own line and reset epoch timer.
        """
        if logs is not None:
            self._last_logs = logs

        steps = self.steps_per_epoch
        bar = self._create_progress_bar(steps, steps)
        step_str = f"{steps}/{steps}" if steps > 0 else "?/?"
        parts = [
            "\r",
            f"{self.name} - Epoch: {epoch + 1}/{self.total_epochs}",
            f"|{bar}| Step: {step_str}",
            "- ETA: 00:00",
        ]

        for name, val in self._last_logs.items():
            parts.append(f"- {name}: {val:.4f}")

        sys.stdout.write(" ".join(parts) + "\n")
        sys.stdout.flush()

        self._reset_epoch_timer()
        
