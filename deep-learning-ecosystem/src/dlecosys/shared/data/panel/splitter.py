"""
Panel Time-Series Splitters
---------------------------
Temporal cross-validation splitters for long-format panel forecasting data.
Unlike tabular CV, panel forecasting splits must happen along the date axis
(never random-row shuffles) to avoid future → past leakage.

Classes
-------
BasePanelSplitter
    Abstract base class exposing ``generate_folds`` yielding
    ``(train_dates, val_dates)`` tuples.

PanelHoldoutSplitter
    Single train/val split at a configured fraction of the sorted unique dates.

PanelExpandingWindowSplitter
    Expanding-window CV. Fold ``k`` trains on all dates up to step ``k`` and
    validates on the next block of dates. Standard time-series CV.

PanelWalkForwardSplitter
    Rolling-window CV. Fold ``k`` trains on a fixed-length window of dates and
    validates on the next block of dates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "BasePanelSplitter",
    "PanelExpandingWindowSplitter",
    "PanelHoldoutSplitter",
    "PanelWalkForwardSplitter",
]


class BasePanelSplitter(ABC):
    """Abstract base class for panel time-series splitters."""

    @abstractmethod
    def generate_folds(
        self, panel: pd.DataFrame, *, date_column: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield ``(train_dates, val_dates)`` tuples over the sorted unique dates.

        Parameters
        ----------
        panel : pandas.DataFrame
            Long-format panel containing ``date_column``.
        date_column : str
            Name of the calendar-date column used for splitting.

        Yields
        ------
        tuple of (numpy.ndarray, numpy.ndarray)
            Sorted train and val dates for one fold.
        """


class PanelHoldoutSplitter(BasePanelSplitter):
    """
    Single train/val split at a configured fraction of sorted unique dates.

    Parameters
    ----------
    val_fraction : float, default=0.2
        Fraction of the trailing sorted unique dates reserved for validation.
    """

    def __init__(self, val_fraction: float = 0.2) -> None:
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must lie strictly between 0 and 1.")
        self.val_fraction = val_fraction

    def generate_folds(
        self, panel: pd.DataFrame, *, date_column: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = _sorted_unique_dates(panel, date_column)
        n = len(dates)
        val_size = max(1, int(n * self.val_fraction))
        train_size = n - val_size
        if train_size <= 0:
            raise ValueError(
                "PanelHoldoutSplitter produced an empty training split; "
                "val_fraction is too large for the available history."
            )
        yield dates[:train_size], dates[train_size:]


class PanelExpandingWindowSplitter(BasePanelSplitter):
    """
    Expanding-window time-series CV.

    For ``n_splits=K`` the ``n`` sorted unique dates are split into ``K + 1``
    equal blocks. Fold ``k`` (zero-based, ``k < K``) trains on all dates in
    blocks ``0 .. k`` and validates on block ``k + 1``.

    Parameters
    ----------
    n_splits : int, default=5
        Number of CV folds.
    """

    def __init__(self, n_splits: int = 5) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        self.n_splits = n_splits

    def generate_folds(
        self, panel: pd.DataFrame, *, date_column: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = _sorted_unique_dates(panel, date_column)
        n = len(dates)
        block_size = n // (self.n_splits + 1)
        if block_size == 0:
            raise ValueError(
                f"Not enough unique dates ({n}) to build {self.n_splits} expanding-window folds."
            )
        for k in range(self.n_splits):
            train_end = block_size * (k + 1)
            val_end = train_end + block_size
            yield dates[:train_end], dates[train_end:val_end]


class PanelWalkForwardSplitter(BasePanelSplitter):
    """
    Rolling-window time-series CV with a fixed-length training window.

    Parameters
    ----------
    n_splits : int, default=5
        Number of CV folds.
    train_window_size : int or None, default=None
        Number of sorted unique dates in the training window for each fold.
        When ``None``, defaults to ``n // (n_splits + 1)``.
    """

    def __init__(self, n_splits: int = 5, train_window_size: int | None = None) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        if train_window_size is not None and train_window_size < 1:
            raise ValueError("train_window_size must be >= 1 when provided.")
        self.n_splits = n_splits
        self.train_window_size = train_window_size

    def generate_folds(
        self, panel: pd.DataFrame, *, date_column: str
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        dates = _sorted_unique_dates(panel, date_column)
        n = len(dates)
        block_size = n // (self.n_splits + 1)
        if block_size == 0:
            raise ValueError(
                f"Not enough unique dates ({n}) to build {self.n_splits} walk-forward folds."
            )
        window_size = self.train_window_size if self.train_window_size is not None else block_size
        for k in range(self.n_splits):
            train_end = block_size * (k + 1)
            train_start = max(0, train_end - window_size)
            val_end = train_end + block_size
            yield dates[train_start:train_end], dates[train_end:val_end]


def _sorted_unique_dates(panel: pd.DataFrame, date_column: str) -> np.ndarray:
    if date_column not in panel.columns:
        raise ValueError(f"Panel is missing required date column {date_column!r}.")
    values = pd.to_datetime(panel[date_column]).dropna().unique()
    if len(values) == 0:
        raise ValueError("Panel contains no valid dates for splitting.")
    return np.sort(values)
