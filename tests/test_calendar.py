"""
Calendar Tests
--------------
Unit tests for market-calendar resolution and long-history fallback behavior.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from density_model.data.calendar import MarketCalendar

__all__ = []


def test_market_calendar_rejects_empty_calendar_name() -> None:
    """
    Reject an empty public calendar identifier.

    Returns
    -------
    None
        This test asserts fail-fast public validation.
    """

    with pytest.raises(ValueError, match="calendar_name must be a non-empty string"):
        MarketCalendar("")


def test_market_calendar_falls_back_to_business_day_backend_when_precise_backend_lacks_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Use the approximate long-history fallback when the precise backend does not
    cover the requested date range.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace backend builders.

    Returns
    -------
    None
        This test asserts fallback session generation and warning behavior.
    """

    class EmptyScheduleCalendar:
        """
        Stub calendar returning no sessions for any request.
        """

        def schedule(self, start_date: date, end_date: date) -> pd.DataFrame:
            return pd.DataFrame(index=pd.DatetimeIndex([]))

    monkeypatch.setattr(MarketCalendar, "_build_pandas_market_calendar", lambda self: EmptyScheduleCalendar())

    calendar = MarketCalendar("XNYS")
    with pytest.warns(RuntimeWarning, match="approximate US business-day calendar logic"):
        sessions = calendar.sessions_in_range(start_date=date(1980, 1, 2), end_date=date(1980, 1, 8))

    assert calendar._fallback_offset is not None
    assert list(sessions) == [
        pd.Timestamp("1980-01-02"),
        pd.Timestamp("1980-01-03"),
        pd.Timestamp("1980-01-04"),
        pd.Timestamp("1980-01-07"),
        pd.Timestamp("1980-01-08"),
    ]

def test_market_calendar_normalizes_nyse_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Normalize common public aliases for the same market calendar.

    Returns
    -------
    None
        This test asserts calendar-name normalization.
    """

    monkeypatch.setattr(MarketCalendar, "_build_pandas_market_calendar", lambda self: object())

    calendar = MarketCalendar("nyse")

    assert calendar.calendar_name == "NYSE"
