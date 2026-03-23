"""
Yahoo Finance Data Tests
------------------------
Unit tests for Yahoo Finance price extraction and daily return transformations.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from density_model.data.yahoo import YahooDailyReturnsLoader, YahooDailyReturnsRequest

__all__ = []


class FakeXNYSCalendar:
    """
    Test double for the XNYS calendar normalizer.

    Parameters
    ----------
    calendar_name : str, default="XNYS"
        Requested calendar identifier.
    """

    def __init__(self, calendar_name: str = "XNYS") -> None:
        self.calendar_name = calendar_name

    def normalize_daily_frame(
        self,
        frame: pd.DataFrame,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Reindex a wide frame onto business-day sessions for tests.

        Parameters
        ----------
        frame : pandas.DataFrame
            Wide daily frame indexed by date.
        start_date : datetime.date
            Inclusive start date.
        end_date : datetime.date
            Inclusive end date.

        Returns
        -------
        pandas.DataFrame
            Business-day reindexed frame used by tests.
        """

        sessions = pd.date_range(start=start_date, end=end_date, freq="B")
        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index).normalize()
        return normalized.reindex(sessions)


def test_request_rejects_lowercase_tickers() -> None:
    """
    Reject ticker symbols that do not already satisfy the public schema.

    Returns
    -------
    None
        This test asserts fail-fast input validation.
    """

    with pytest.raises(ValidationError, match="Ticker symbols must be uppercase"):
        YahooDailyReturnsRequest(
            tickers=("spy", "QQQ"),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
        )


def test_request_rejects_empty_symbol_list() -> None:
    """
    Reject a request that does not contain any valid ticker symbols.

    Returns
    -------
    None
        This test asserts that invalid input raises ``ValueError``.
    """

    with pytest.raises(ValidationError, match="At least one ticker symbol is required"):
        YahooDailyReturnsRequest(
            tickers=(),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
        )


def test_request_accepts_non_default_calendar_alias() -> None:
    """
    Accept a non-default market calendar identifier when it satisfies the
    public schema.

    Returns
    -------
    None
        This test asserts non-empty calendar support.
    """

    request = YahooDailyReturnsRequest(
        tickers=("SPY",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        calendar="nyse",
    )

    assert request.calendar == "NYSE"


def test_load_returns_computes_simple_daily_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Compute simple daily returns from downloaded adjusted close prices.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace the external Yahoo Finance downloader.

    Returns
    -------
    None
        This test asserts the returned daily simple returns.
    """

    def fake_download(**_: object) -> pd.DataFrame:
        columns = pd.MultiIndex.from_product([["Adj Close"], ["SPY", "QQQ"]])
        return pd.DataFrame(
            [[100.0, 200.0], [110.0, 180.0], [121.0, 198.0]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            columns=columns,
        )

    import density_model.data.yahoo as yahoo_module

    monkeypatch.setattr(yahoo_module, "yf", type("FakeYFinance", (), {"download": staticmethod(fake_download)}))
    monkeypatch.setattr(yahoo_module, "XNYSCalendar", FakeXNYSCalendar)

    loader = YahooDailyReturnsLoader()
    request = YahooDailyReturnsRequest(
        tickers=("SPY", "QQQ"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
        drop_missing=True,
    )

    result = loader.load_returns(request)
    expected = pd.DataFrame(
        {"SPY": [0.10, 0.10], "QQQ": [-0.10, 0.10]},
        index=pd.date_range("2024-01-03", periods=2, freq="B"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_load_prices_rejects_missing_price_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Raise an error when the requested Yahoo Finance price field is absent.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace the external Yahoo Finance downloader.

    Returns
    -------
    None
        This test asserts that invalid payload structure raises ``ValueError``.
    """

    def fake_download(**_: object) -> pd.DataFrame:
        columns = pd.MultiIndex.from_product([["Close"], ["SPY"]])
        return pd.DataFrame(
            [[100.0], [101.0]],
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            columns=columns,
        )

    import density_model.data.yahoo as yahoo_module

    monkeypatch.setattr(yahoo_module, "yf", type("FakeYFinance", (), {"download": staticmethod(fake_download)}))
    monkeypatch.setattr(yahoo_module, "XNYSCalendar", FakeXNYSCalendar)

    loader = YahooDailyReturnsLoader()
    request = YahooDailyReturnsRequest(
        tickers=("SPY",),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 2, 1),
    )

    with pytest.raises(ValueError, match="Price field 'Adj Close' not found"):
        loader.load_prices(request)


def test_load_prices_normalizes_to_xnys_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Reindex downloaded prices to the XNYS master session calendar.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace the external Yahoo Finance downloader.

    Returns
    -------
    None
        This test asserts calendar-normalized daily prices.
    """

    def fake_download(**_: object) -> pd.DataFrame:
        columns = pd.MultiIndex.from_product([["Adj Close"], ["SPY"]])
        return pd.DataFrame(
            [[100.0], [102.0]],
            index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
            columns=columns,
        )

    import density_model.data.yahoo as yahoo_module

    monkeypatch.setattr(yahoo_module, "yf", type("FakeYFinance", (), {"download": staticmethod(fake_download)}))
    monkeypatch.setattr(yahoo_module, "XNYSCalendar", FakeXNYSCalendar)

    loader = YahooDailyReturnsLoader()
    request = YahooDailyReturnsRequest(
        tickers=("SPY",),
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 5),
    )

    result = loader.load_prices(request)

    expected = pd.DataFrame(
        {"SPY": [100.0, pd.NA, 102.0, pd.NA]},
        index=pd.date_range("2024-01-02", periods=4, freq="B"),
        dtype="object",
    )
    expected["SPY"] = pd.to_numeric(expected["SPY"], errors="coerce")
    pd.testing.assert_frame_equal(result, expected)


def test_calendar_normalization_rejects_dates_outside_supported_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Raise a clear error when the requested date range predates calendar support.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture used to replace the external Yahoo Finance downloader.

    Returns
    -------
    None
        This test asserts explicit calendar-boundary validation.
    """

    def fake_download(**_: object) -> pd.DataFrame:
        columns = pd.MultiIndex.from_product([["Adj Close"], ["SPY"]])
        return pd.DataFrame(
            [[100.0], [102.0]],
            index=pd.to_datetime(["2000-01-03", "2000-01-04"]),
            columns=columns,
        )

    class BoundedCalendar:
        """
        Test double for a market calendar that rejects unsupported historical
        ranges.

        Parameters
        ----------
        calendar_name : str, default="XNYS"
            Requested calendar identifier.
        """

        def __init__(self, calendar_name: str = "XNYS") -> None:
            self.calendar_name = calendar_name

        def normalize_daily_frame(
            self,
            frame: pd.DataFrame,
            *,
            start_date: date,
            end_date: date,
        ) -> pd.DataFrame:
            """
            Raise when the requested range falls outside the supported backend.
            """

            if start_date < date(2006, 3, 22):
                raise ValueError(
                    "Requested start_date 2000-01-01 falls outside the supported "
                    "calendar backend range."
                )
            return frame

    import density_model.data.yahoo as yahoo_module

    monkeypatch.setattr(yahoo_module, "yf", type("FakeYFinance", (), {"download": staticmethod(fake_download)}))
    monkeypatch.setattr(yahoo_module, "XNYSCalendar", BoundedCalendar)

    loader = YahooDailyReturnsLoader()
    request = YahooDailyReturnsRequest(
        tickers=("SPY",),
        start_date=date(2000, 1, 1),
        end_date=date(2000, 1, 31),
    )

    with pytest.raises(ValueError, match="supported calendar backend range"):
        loader.load_prices(request)
