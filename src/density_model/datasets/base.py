"""
Base Dataset
------------
Shared base class for canonicalized panel time-series datasets.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from density_model.datasets.config import TimeSeriesConfig
from density_model.datasets.inspectors import TimeSeriesInspector

__all__ = ["BaseTimeSeriesDataset"]


class BaseTimeSeriesDataset:
    """
    Base class for canonicalized panel time-series datasets.

    Parameters
    ----------
    data : Any
        Tabular input data.
    date_column : str
        Date column name.
    id_column : str
        Entity identifier column name.
    coerce_dates : bool, default=False
        Whether to coerce the date column to datetime.
    sort : bool, default=True
        Whether to sort by identifier and date.
    """

    __slots__ = ("_data", "time_series_cfg")

    def __init__(
        self,
        *,
        data: Any,
        date_column: str,
        id_column: str,
        coerce_dates: bool = False,
        sort: bool = True,
    ) -> None:
        self.time_series_cfg = TimeSeriesConfig(
            date_column=date_column,
            id_column=id_column,
            coerce_dates=coerce_dates,
            sort=sort,
        )
        self._data = TimeSeriesInspector.canonicalize(
            data,
            date_column=self.time_series_cfg.date_column,
            id_column=self.time_series_cfg.id_column,
            coerce_dates=self.time_series_cfg.coerce_dates,
            sort=self.time_series_cfg.sort,
        )

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the canonicalized dataset.

        Returns
        -------
        pandas.DataFrame
            Canonicalized panel DataFrame.
        """

        return self._data

    @property
    def date_column(self) -> str:
        """
        Return the configured date column name.

        Returns
        -------
        str
            Date column name.
        """

        return self.time_series_cfg.date_column

    @property
    def id_column(self) -> str:
        """
        Return the configured identifier column name.

        Returns
        -------
        str
            Identifier column name.
        """

        return self.time_series_cfg.id_column
