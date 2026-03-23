"""
Dataset Inspectors
------------------
Validation helpers for canonicalizing panel time-series inputs and checking
forecasting-specific structural requirements.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["FrameBackend", "ForecastingInspector", "TimeSeriesInspector"]


class FrameBackend:
    """
    Normalize supported tabular inputs to pandas objects.

    Parameters
    ----------
    None
        This class exposes only normalization helpers.
    """

    @staticmethod
    def to_pandas(obj: Any) -> pd.DataFrame:
        """
        Convert a supported tabular object to a pandas DataFrame.

        Parameters
        ----------
        obj : Any
            Object expected to be a pandas or polars DataFrame or Series.

        Returns
        -------
        pandas.DataFrame
            Canonical pandas DataFrame representation.

        Raises
        ------
        TypeError
            If the provided object is not a supported tabular type.
        """

        if isinstance(obj, pd.DataFrame):
            return obj
        if isinstance(obj, pd.Series):
            name = obj.name if obj.name is not None else "value"
            return obj.to_frame(name)
        try:
            import polars as pl  # type: ignore
        except ImportError:
            pl = None
        if pl is not None and isinstance(obj, pl.DataFrame):
            return obj.to_pandas()
        if pl is not None and isinstance(obj, pl.Series):
            name = obj.name if obj.name is not None else "value"
            return obj.to_pandas().to_frame(name)
        raise TypeError("Expected a pandas or polars DataFrame or Series.")


class TimeSeriesInspector:
    """
    Validate and canonicalize panel time-series inputs.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """

    @staticmethod
    def canonicalize(
        obj: Any,
        *,
        date_column: str,
        id_column: str,
        coerce_dates: bool,
        sort: bool,
    ) -> pd.DataFrame:
        """
        Canonicalize tabular input for panel forecasting datasets.

        Parameters
        ----------
        obj : Any
            Tabular data object to canonicalize.
        date_column : str
            Date column name.
        id_column : str
            Entity identifier column name.
        coerce_dates : bool
            Whether to coerce the date column to datetime.
        sort : bool
            Whether to sort by identifier and date.

        Returns
        -------
        pandas.DataFrame
            Canonicalized panel DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing or identifier-date pairs are duplicated.
        TypeError
            If the date column is not datetime-like after optional coercion.
        """

        frame = FrameBackend.to_pandas(obj)
        if date_column not in frame.columns:
            raise ValueError(f"Required date column {date_column!r} was not found.")
        if id_column not in frame.columns:
            raise ValueError(f"Required id column {id_column!r} was not found.")
        if coerce_dates:
            frame = frame.copy()
            frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
        if not pd.api.types.is_datetime64_any_dtype(frame[date_column]):
            raise TypeError(f"Column {date_column!r} must be datetime-like.")
        if frame.duplicated(subset=[id_column, date_column]).any():
            raise ValueError("Each (id, date) pair must be unique for panel forecasting datasets.")
        if sort:
            frame = frame.sort_values([id_column, date_column]).reset_index(drop=True)
        return frame

    @staticmethod
    def validate_static_features(df: pd.DataFrame, *, id_column: str, static_features: tuple[str, ...]) -> None:
        """
        Validate that declared static features are constant within each identifier.

        Parameters
        ----------
        df : pandas.DataFrame
            Canonicalized panel DataFrame.
        id_column : str
            Entity identifier column name.
        static_features : tuple of str
            Static feature column names.

        Returns
        -------
        None
            This method raises when validation fails.

        Raises
        ------
        ValueError
            If a static feature changes within an identifier group.
        """

        for column in static_features:
            nunique = df.groupby(id_column, sort=False)[column].nunique(dropna=False)
            if (nunique > 1).any():
                raise ValueError(f"Static feature {column!r} must be constant within each id.")


class ForecastingInspector:
    """
    Validate forecasting-window requirements for dataset construction.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """

    @staticmethod
    def validate_window_lengths(*, in_steps: int, horizon: int, out_steps: int) -> None:
        """
        Validate forecasting window lengths.

        Parameters
        ----------
        in_steps : int
            Number of historical input steps.
        horizon : int
            Positive forecasting horizon.
        out_steps : int
            Number of target steps to predict.

        Returns
        -------
        None
            This method raises when validation fails.
        """

        if in_steps <= 0:
            raise ValueError("in_steps must be strictly positive.")
        if horizon <= 0:
            raise ValueError("horizon must be strictly positive.")
        if out_steps <= 0:
            raise ValueError("out_steps must be strictly positive.")
