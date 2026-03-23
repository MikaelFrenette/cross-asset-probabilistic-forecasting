"""
Training Utilities
------------------
Progress display and metric tracking utilities used by the base training loop.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch

__all__ = ["LogRow", "MetricsTracker", "ProgressBar"]


@dataclass(frozen=True, slots=True)
class LogRow:
    """
    Immutable metric snapshot recorded by the training logger.

    Parameters
    ----------
    step : int or None
        Training step associated with the snapshot.
    epoch : int or None
        Epoch associated with the snapshot.
    metrics : dict of str to float
        Scalar metric values stored for the snapshot.
    """

    step: int | None
    epoch: int | None
    metrics: dict[str, float]


class MetricsTracker:
    """
    Track current and historical scalar metrics during training.

    Parameters
    ----------
    None
        Metric state is managed internally.
    """

    def __init__(self) -> None:
        self._current: dict[str, float] = {}
        self._rows: list[LogRow] = []

    def _to_float(self, value: Any, *, name: str) -> float:
        """
        Convert a metric value to a finite Python float.

        Parameters
        ----------
        value : Any
            Metric value to convert.
        name : str
            Metric name used in validation messages.

        Returns
        -------
        float
            Finite scalar float value.

        Raises
        ------
        TypeError
            If the value is not a supported scalar type.
        ValueError
            If the value is not finite or is a non-scalar tensor.
        """

        if isinstance(value, bool):
            raise TypeError(f"Metric {name!r} must be numeric, not bool.")
        if isinstance(value, (int, float)):
            scalar = float(value)
        elif torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Metric {name!r} must be scalar; got tensor with numel={value.numel()}.")
            scalar = float(value.detach().item())
        else:
            raise TypeError(f"Metric {name!r} must be int, float, or scalar torch.Tensor.")
        if not math.isfinite(scalar):
            raise ValueError(f"Metric {name!r} must be finite; got {scalar!r}.")
        return scalar

    def reset_state(self) -> None:
        """
        Clear current metrics and history.

        Returns
        -------
        None
            This method resets all logger state.
        """

        self._current.clear()
        self._rows.clear()

    def clear_current(self) -> None:
        """
        Clear only the current metric state.

        Returns
        -------
        None
            This method preserves metric history.
        """

        self._current.clear()

    def update_state(self, values: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """
        Update the current metric state.

        Parameters
        ----------
        values : dict of str to Any or None, default=None
            Mapping of metric names to values.
        **kwargs : Any
            Additional metric values merged into the update.

        Returns
        -------
        None
            This method updates the current metric snapshot.
        """

        merged: dict[str, Any] = {}
        if values is not None:
            merged.update(values)
        merged.update(kwargs)
        for key, value in merged.items():
            if not isinstance(key, str) or not key:
                raise TypeError("Metric names must be non-empty strings.")
            self._current[key] = self._to_float(value, name=key)

    def last_log(self) -> dict[str, float]:
        """
        Return the current metric state.

        Returns
        -------
        dict of str to float
            Flat mapping of metric names to scalar values.
        """

        return dict(self._current)

    def push(self, *, epoch: int | None = None, step: int | None = None) -> None:
        """
        Append the current metric state to history.

        Parameters
        ----------
        epoch : int or None, default=None
            Epoch associated with the history row.
        step : int or None, default=None
            Step associated with the history row.

        Returns
        -------
        None
            This method appends a new history row.
        """

        self._rows.append(LogRow(step=step, epoch=epoch, metrics=dict(self._current)))

    def summary(self) -> dict[str, dict[str, float]]:
        """
        Summarize metric history with min, max, and average values.

        Returns
        -------
        dict of str to dict of str to float
            Metric summaries keyed by metric name.
        """

        series: dict[str, list[float]] = {}
        for row in self._rows:
            for key, value in row.metrics.items():
                series.setdefault(key, []).append(value)
        return {
            key: {
                "min": float(min(values)),
                "max": float(max(values)),
                "avg": float(sum(values) / len(values)),
            }
            for key, values in series.items()
            if values
        }

    @property
    def history(self) -> pd.DataFrame:
        """
        Return metric history as a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per ``push`` call.

        Raises
        ------
        ValueError
            If no history rows have been recorded.
        """

        if not self._rows:
            raise ValueError("No history available. Call `push(...)` at least once.")
        rows: list[dict[str, Any]] = []
        for row in self._rows:
            payload = {"epoch": row.epoch, "step": row.step}
            payload.update(row.metrics)
            rows.append(payload)
        return pd.DataFrame(rows)


class ProgressBar:
    """
    Display a lightweight console progress bar for epoch-based training loops.

    Parameters
    ----------
    name : str
        Label shown at the beginning of the progress line.
    total_epochs : int
        Total number of training epochs.
    steps_per_epoch : int
        Number of steps per epoch.
    length : int, default=20
        Width of the progress bar.
    fill : str, default="▄"
        Character used for the filled portion of the bar.
    eta_smoothing : float, default=0.2
        Exponential smoothing factor used for ETA estimation.
    """

    def __init__(
        self,
        name: str,
        total_epochs: int,
        steps_per_epoch: int,
        length: int = 20,
        fill: str = "▄",
        eta_smoothing: float = 0.2,
    ) -> None:
        self.name = name
        self.total_epochs = int(total_epochs)
        self.steps_per_epoch = int(steps_per_epoch) if steps_per_epoch is not None else 0
        self.length = int(length)
        self.fill = fill
        self.eta_smoothing = float(eta_smoothing)
        self.start_time: float | None = None
        self.last_logs: dict[str, float] = {}
        self.ema_step_time: float | None = None
        self.use_ansi = sys.stdout.isatty()
        self._last_render_width = 0

    def _format_eta(self, eta_seconds: int) -> str:
        """
        Format ETA seconds as a display string.

        Parameters
        ----------
        eta_seconds : int
            Estimated seconds remaining.

        Returns
        -------
        str
            Formatted ``HH:MM:SS`` or ``MM:SS`` ETA string.
        """

        eta_seconds = max(int(eta_seconds), 0)
        hours, remainder = divmod(eta_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        return f"{minutes:02}:{seconds:02}"

    def _create_progress_bar(self, step: int, steps_per_epoch: int) -> str:
        """
        Create the visual progress bar string.

        Parameters
        ----------
        step : int
            Current step index.
        steps_per_epoch : int
            Total number of steps per epoch.

        Returns
        -------
        str
            Rendered progress bar.
        """

        if steps_per_epoch <= 0:
            return "-" * self.length
        bounded_step = max(0, min(int(step), int(steps_per_epoch)))
        filled_length = int(self.length * bounded_step // steps_per_epoch)
        filled = self.fill * filled_length
        empty = "-" * (self.length - filled_length)
        if self.use_ansi:
            return f"\033[92m{filled}\033[0m{empty}"
        return f"{filled}{empty}"

    def _reset_epoch_timer(self) -> None:
        """
        Reset epoch-local timing state.

        Returns
        -------
        None
            This method resets ETA estimation.
        """

        self.start_time = time.time()
        self.ema_step_time = None

    def __call__(self, epoch: int, step: int, logs: dict[str, float]) -> None:
        """
        Update the progress display for the current step.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        step : int
            One-based step index.
        logs : dict of str to float
            Current aggregated metrics.

        Returns
        -------
        None
            This method writes the progress line to stdout.
        """

        if self.start_time is None:
            self._reset_epoch_timer()

        self.last_logs = logs or {}
        if self.steps_per_epoch <= 0:
            eta_formatted = "--:--"
            time_text = "--"
            bar = self._create_progress_bar(0, 0)
            step_display = int(step)
            steps_display: int | str = "?"
        else:
            step_index = max(1, min(int(step), self.steps_per_epoch))
            elapsed = time.time() - float(self.start_time)
            avg_step_time = elapsed / max(step_index, 1)
            if self.ema_step_time is None:
                self.ema_step_time = avg_step_time
            else:
                alpha = self.eta_smoothing
                self.ema_step_time = alpha * avg_step_time + (1.0 - alpha) * self.ema_step_time
            steps_left = self.steps_per_epoch - step_index
            eta_formatted = self._format_eta(int(max(steps_left, 0) * float(self.ema_step_time)))
            time_text = (
                f"{self.ema_step_time * 1000.0:.0f}ms/step"
                if self.ema_step_time < 1.0
                else f"{self.ema_step_time:.2f}s/step"
            )
            bar = self._create_progress_bar(step_index, self.steps_per_epoch)
            step_display = step_index
            steps_display = self.steps_per_epoch

        parts = [
            f"{self.name} - Epoch: {epoch + 1}/{self.total_epochs}",
            f"|{bar}| Step: {step_display}/{steps_display}",
            f"- ETA: {eta_formatted}",
            f"- {time_text}",
        ]
        for name, value in self.last_logs.items():
            parts.append(f"- {name}: {value:.3f}")
        self._write_line(" ".join(parts))

    def end_epoch(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        """
        Finalize the epoch progress line and reset epoch timing state.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict of str to float or None, default=None
            Current aggregated metrics.

        Returns
        -------
        None
            This method writes a completed progress line to stdout.
        """

        if logs is not None:
            self.last_logs = logs
        bar = self._create_progress_bar(self.steps_per_epoch, self.steps_per_epoch)
        parts = [
            f"{self.name} - Epoch: {epoch + 1}/{self.total_epochs}",
            f"|{bar}| Step: {self.steps_per_epoch}/{self.steps_per_epoch}",
            "- ETA: 00:00",
        ]
        for name, value in self.last_logs.items():
            parts.append(f"- {name}: {value:.3f}")
        self._write_line(" ".join(parts), final=True)
        self._reset_epoch_timer()

    def _write_line(self, message: str, *, final: bool = False) -> None:
        """
        Render one progress line while clearing any leftover characters.

        Parameters
        ----------
        message : str
            Progress text to render.
        final : bool, default=False
            Whether to terminate the line with a newline.

        Returns
        -------
        None
            This method writes the progress line to stdout.
        """

        if self.use_ansi:
            terminator = "\n" if final else ""
            sys.stdout.write(f"\r{message}\033[K{terminator}")
        else:
            padded = message
            if len(message) < self._last_render_width:
                padded = message + (" " * (self._last_render_width - len(message)))
            terminator = "\n" if final else ""
            sys.stdout.write(f"\r{padded}{terminator}")
        sys.stdout.flush()
        self._last_render_width = 0 if final else len(message)
