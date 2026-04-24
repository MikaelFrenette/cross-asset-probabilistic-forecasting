"""
Yahoo Return Panel Sources
--------------------------
Download calendar-normalized Yahoo Finance daily returns and assemble canonical
long-format return panels for volatility forecasting workflows.

Classes
-------
YahooDailyReturnsRequest
    Pydantic request object for Yahoo Finance downloads (ticker set, date
    range, price field, master calendar, missing-value policy).

YahooDailyReturnsLoader
    Download adjusted close prices and convert them to daily simple returns,
    reindexed onto a master trading calendar.

YahooVolatilityPanelBuilder
    Build canonical long-format ``(date, asset_id, return, ticker)`` panels
    from calendar-normalized Yahoo returns for panel forecasting models.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dlecosys.shared.data.panel.calendar import XNYSCalendar

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    yf = None

__all__ = [
    "YahooDailyReturnsLoader",
    "YahooDailyReturnsRequest",
    "YahooVolatilityPanelBuilder",
]


class YahooDailyReturnsRequest(BaseModel):
    """Request definition for downloading Yahoo Finance daily returns."""

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
        if not value:
            raise ValueError("At least one ticker symbol is required.")
        for ticker in value:
            if not ticker or ticker.strip() != ticker:
                raise ValueError(
                    "Ticker symbols must be non-empty and must not include surrounding whitespace."
                )
            if ticker != ticker.upper():
                raise ValueError("Ticker symbols must be uppercase.")
        return value

    @field_validator("price_field")
    @classmethod
    def validate_price_field(cls, value: str) -> str:
        if value != "Adj Close":
            raise ValueError("Only 'Adj Close' is currently supported.")
        return value

    @model_validator(mode="after")
    def validate_date_order(self) -> YahooDailyReturnsRequest:
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be earlier than end_date.")
        return self

    @field_validator("calendar")
    @classmethod
    def validate_calendar(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("calendar must be a non-empty string.")
        return normalized


class YahooDailyReturnsLoader:
    """Loader for Yahoo Finance adjusted close prices and daily returns."""

    def __init__(self, auto_adjust: bool = False, progress: bool = False) -> None:
        self.auto_adjust = auto_adjust
        self.progress = progress

    def load_prices(self, request: YahooDailyReturnsRequest) -> pd.DataFrame:
        """Download daily prices for the requested ticker set."""

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
        """Download prices and convert them to simple daily returns."""

        prices = self.load_prices(request=request)
        returns = prices.pct_change(fill_method=None)
        if request.drop_missing:
            returns = returns.dropna(how="any")
        return returns

    def _calendarize_prices(
        self, prices: pd.DataFrame, request: YahooDailyReturnsRequest
    ) -> pd.DataFrame:
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
        if downloaded.empty:
            return pd.DataFrame(columns=list(tickers), dtype=float)

        if isinstance(downloaded.columns, pd.MultiIndex):
            if request.price_field not in downloaded.columns.get_level_values(0):
                raise ValueError(
                    f"Price field '{request.price_field}' not found in Yahoo Finance data."
                )
            prices = downloaded[request.price_field]
        else:
            if request.price_field not in downloaded.columns:
                raise ValueError(
                    f"Price field '{request.price_field}' not found in Yahoo Finance data."
                )
            prices = downloaded[[request.price_field]].rename(
                columns={request.price_field: tickers[0]}
            )

        missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
        for ticker in missing_tickers:
            prices[ticker] = pd.NA

        return prices.loc[:, list(tickers)]


class YahooVolatilityPanelBuilder:
    """Build a canonical panel DataFrame for return forecasting."""

    def __init__(
        self,
        id_column: str = "asset_id",
        date_column: str = "date",
        return_column: str = "return",
        ticker_column: str = "ticker",
    ) -> None:
        self.id_column = id_column
        self.date_column = date_column
        self.return_column = return_column
        self.ticker_column = ticker_column

    def build_from_loader(
        self,
        *,
        loader: YahooDailyReturnsLoader,
        request: YahooDailyReturnsRequest,
    ) -> pd.DataFrame:
        """Download returns and build the canonical forecasting panel."""

        returns = loader.load_returns(request=request)
        return self.build_from_returns(returns=returns)

    def build_from_returns(self, *, returns: pd.DataFrame) -> pd.DataFrame:
        """Convert wide returns into a canonical long-format forecasting panel."""

        return_frame = returns.stack(future_stack=True).rename(self.return_column).reset_index()
        return_frame.columns = [self.date_column, self.id_column, self.return_column]
        return_frame[self.ticker_column] = return_frame[self.id_column]
        return return_frame.sort_values([self.id_column, self.date_column]).reset_index(drop=True)

    def transform_feature_panel(self, *, panel: pd.DataFrame) -> pd.DataFrame:
        """Validate and canonicalize a pre-existing long-format feature panel."""

        required_columns = {self.date_column, self.id_column, self.return_column}
        missing_columns = sorted(required_columns.difference(panel.columns))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Feature panel is missing required columns: {missing_text}")
        return panel.sort_values([self.id_column, self.date_column]).reset_index(drop=True)
