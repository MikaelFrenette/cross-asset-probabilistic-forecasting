"""
Yahoo Finance Daily Returns
---------------------------
Download adjusted close prices from Yahoo Finance and transform them into daily returns
for a specified subset of equity tickers.
"""

from __future__ import annotations

from datetime import date
from datetime import timedelta

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from density_model.data.calendar import XNYSCalendar

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    yf = None

__all__ = ["YahooDailyReturnsLoader", "YahooDailyReturnsRequest"]


class YahooDailyReturnsRequest(BaseModel):
    """
    Request definition for downloading Yahoo Finance daily returns.

    Parameters
    ----------
    tickers : tuple of str
        Sequence of Yahoo Finance ticker symbols to download.
    start_date : datetime.date
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date : datetime.date
        Inclusive end date in ``YYYY-MM-DD`` format.
    price_field : str, default="Adj Close"
        Price field used to compute returns.
    calendar : str, default="XNYS"
        Master session calendar used to normalize daily prices.
    drop_missing : bool, default=False
        Whether to drop rows with missing returns after transformation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tickers: tuple[str, ...]
    start_date: date
    end_date: date
    price_field: str = "Adj Close"
    calendar: str = "XNYS"
    drop_missing: bool = False

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """
        Validate requested ticker symbols.

        Parameters
        ----------
        value : tuple of str
            Raw ticker sequence.

        Returns
        -------
        tuple of str
            Validated ticker sequence.
        """

        if not value:
            raise ValueError("At least one ticker symbol is required.")
        for ticker in value:
            if not ticker or ticker.strip() != ticker:
                raise ValueError("Ticker symbols must be non-empty and must not include surrounding whitespace.")
            if ticker != ticker.upper():
                raise ValueError("Ticker symbols must be uppercase.")
        return value

    @field_validator("price_field")
    @classmethod
    def validate_price_field(cls, value: str) -> str:
        """
        Validate the requested price field.

        Parameters
        ----------
        value : str
            Requested Yahoo Finance price field.

        Returns
        -------
        str
            Validated price field name.
        """

        if value != "Adj Close":
            raise ValueError("Only 'Adj Close' is currently supported.")
        return value

    @model_validator(mode="after")
    def validate_date_order(self) -> YahooDailyReturnsRequest:
        """
        Validate the configured date range.

        Returns
        -------
        YahooDailyReturnsRequest
            Validated request object.
        """

        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date.")
        return self

    @field_validator("calendar")
    @classmethod
    def validate_calendar(cls, value: str) -> str:
        """
        Validate the configured master calendar.

        Parameters
        ----------
        value : str
            Requested master calendar identifier.

        Returns
        -------
        str
            Validated master calendar identifier.
        """

        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("calendar must be a non-empty string.")
        return normalized


class YahooDailyReturnsLoader:
    """
    Loader for Yahoo Finance adjusted close prices and daily returns.

    Parameters
    ----------
    auto_adjust : bool, default=False
        Whether Yahoo Finance should auto-adjust downloaded prices.
    progress : bool, default=False
        Whether to show the download progress bar.
    """

    def __init__(self, auto_adjust: bool = False, progress: bool = False) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress

    def load_prices(self, request: YahooDailyReturnsRequest) -> pd.DataFrame:
        """
        Download daily prices for the requested ticker set.

        Parameters
        ----------
        request : YahooDailyReturnsRequest
            Download specification for ticker symbols and date bounds.

        Returns
        -------
        pandas.DataFrame
            Wide price matrix indexed by trading date with one column per ticker.

        Raises
        ------
        ImportError
            If ``yfinance`` is not installed.
        ValueError
            If the requested price field is unavailable in the downloaded payload.
        """

        if yf is None:
            raise ImportError("yfinance must be installed to download Yahoo Finance data.")

        tickers = request.tickers
        downloaded = yf.download(
            tickers=list(tickers),
            start=request.start_date.isoformat(),
            end=(request.end_date + timedelta(days=1)).isoformat(),
            auto_adjust=self.auto_adjust,
            progress=self.progress,
        )
        prices = self._extract_price_frame(downloaded=downloaded, request=request, tickers=tickers)
        prices.index = pd.to_datetime(prices.index)
        prices.columns = [str(column).upper() for column in prices.columns]
        prices = prices.sort_index()
        return self._calendarize_prices(prices=prices, request=request)

    def load_returns(self, request: YahooDailyReturnsRequest) -> pd.DataFrame:
        """
        Download prices and convert them to simple daily returns.

        Parameters
        ----------
        request : YahooDailyReturnsRequest
            Download specification for ticker symbols and date bounds.

        Returns
        -------
        pandas.DataFrame
            Daily simple returns indexed by trading date.
        """

        prices = self.load_prices(request=request)
        returns = prices.pct_change(fill_method=None)
        if request.drop_missing:
            returns = returns.dropna(how="any")
        return returns

    def _calendarize_prices(self, prices: pd.DataFrame, request: YahooDailyReturnsRequest) -> pd.DataFrame:
        """
        Normalize downloaded prices to the configured master trading calendar.

        Parameters
        ----------
        prices : pandas.DataFrame
            Wide price matrix indexed by observed trading dates.
        request : YahooDailyReturnsRequest
            Download request containing the master calendar specification.

        Returns
        -------
        pandas.DataFrame
            Wide price matrix reindexed to the configured master calendar.
        """

        calendar = XNYSCalendar(calendar_name=request.calendar)
        return calendar.normalize_daily_frame(
            prices,
            start_date=request.start_date,
            end_date=request.end_date,
        )

    def _extract_price_frame(
        self,
        downloaded: pd.DataFrame,
        request: YahooDailyReturnsRequest,
        tickers: tuple[str, ...],
    ) -> pd.DataFrame:
        """
        Extract a wide price matrix from a Yahoo Finance download payload.

        Parameters
        ----------
        downloaded : pandas.DataFrame
            Raw payload returned by ``yfinance.download``.
        request : YahooDailyReturnsRequest
            Download specification including the required price field.
        tickers : tuple of str
            Normalized ticker symbols requested by the caller.

        Returns
        -------
        pandas.DataFrame
            Wide price matrix with one column per ticker.

        Raises
        ------
        ValueError
            If the required price field cannot be extracted.
        """

        if downloaded.empty:
            return pd.DataFrame(columns=list(tickers), dtype=float)

        if isinstance(downloaded.columns, pd.MultiIndex):
            if request.price_field not in downloaded.columns.get_level_values(0):
                raise ValueError(f"Price field '{request.price_field}' not found in Yahoo Finance data.")
            prices = downloaded[request.price_field]
        else:
            if request.price_field not in downloaded.columns:
                raise ValueError(f"Price field '{request.price_field}' not found in Yahoo Finance data.")
            prices = downloaded[[request.price_field]].rename(columns={request.price_field: tickers[0]})

        missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
        for ticker in missing_tickers:
            prices[ticker] = pd.NA

        return prices.loc[:, list(tickers)]
