"""
Market Calendar
---------------
Daily market-session utilities built around ``pandas_market_calendars`` with a
lightweight internal fallback for long-history US equity workflows.

Classes
-------
MarketCalendar
    Generic daily market calendar with ``pandas_market_calendars`` backend and
    US business-day fallback.

XNYSCalendar
    Backward-compatible subclass that defaults to the ``XNYS`` calendar alias.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

__all__ = ["MarketCalendar", "XNYSCalendar"]


class MarketCalendar:
    """Generic daily market calendar helper."""

    def __init__(self, calendar_name: str = "XNYS") -> None:
        self.calendar_name = self._normalize_calendar_name(calendar_name)
        self._calendar = self._build_pandas_market_calendar()
        self._fallback_offset: CustomBusinessDay | None = None

    def sessions_in_range(self, start_date: date, end_date: date) -> pd.DatetimeIndex:
        """Return master trading sessions in an inclusive date range."""

        if start_date > end_date:
            raise ValueError("start_date must be earlier than or equal to end_date.")
        if self._calendar is not None:
            schedule = self._calendar.schedule(start_date=start_date, end_date=end_date)
            if not schedule.empty:
                return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
            self._activate_us_equity_fallback()
        if self._fallback_offset is None:
            raise RuntimeError(f"Unable to resolve sessions for {self.calendar_name!r}.")
        return pd.date_range(start=start_date, end=end_date, freq=self._fallback_offset)

    def next_session_after(self, session_date: date | pd.Timestamp) -> pd.Timestamp:
        """Return the first session strictly after a given session date."""

        normalized_date = pd.Timestamp(session_date).tz_localize(None).normalize()
        if self._calendar is not None:
            search_start = normalized_date + timedelta(days=1)
            search_end = normalized_date + timedelta(days=31)
            schedule = self._calendar.schedule(
                start_date=search_start.date(), end_date=search_end.date()
            )
            if not schedule.empty:
                return pd.Timestamp(schedule.index[0]).tz_localize(None).normalize()
            self._activate_us_equity_fallback()
        if self._fallback_offset is None:
            raise RuntimeError(f"Unable to resolve a next session for {self.calendar_name!r}.")
        next_values = pd.date_range(
            start=normalized_date + timedelta(days=1), periods=1, freq=self._fallback_offset
        )
        if len(next_values) != 1:
            raise ValueError(
                f"Unable to locate a future session after {normalized_date.date().isoformat()}."
            )
        return pd.Timestamp(next_values[0]).normalize()

    def normalize_daily_frame(
        self,
        frame: pd.DataFrame,
        *,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Reindex a daily wide matrix onto the master session calendar."""

        normalized = frame.copy()
        normalized.index = pd.to_datetime(normalized.index).tz_localize(None).normalize()
        normalized = normalized.sort_index()
        sessions = self.sessions_in_range(start_date=start_date, end_date=end_date)
        return normalized.reindex(sessions)

    def _build_pandas_market_calendar(self):  # type: ignore[no-untyped-def]
        try:
            import pandas_market_calendars as pmc
        except ImportError as error:
            raise ImportError(
                "pandas_market_calendars must be installed to use market calendar normalization."
            ) from error

        for alias in self._candidate_backend_names():
            try:
                return pmc.get_calendar(alias)
            except Exception:
                continue
        return None

    def _activate_us_equity_fallback(self) -> None:
        if self._fallback_offset is not None:
            return
        supported_aliases = {"XNYS", "NYSE", "XNAS", "NASDAQ"}
        if self.calendar_name not in supported_aliases:
            raise ValueError(
                f"pandas_market_calendars does not expose a calendar matching "
                f"{self.calendar_name!r}, and no internal fallback exists for this market."
            )
        self._fallback_offset = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        warnings.warn(
            (
                f"Falling back to approximate US business-day calendar logic for "
                f"{self.calendar_name}. This is suitable for daily workflows but may not "
                "match historical exchange closures exactly."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    def _candidate_backend_names(self) -> tuple[str, ...]:
        alias_map = {
            "XNYS": ("XNYS", "NYSE"),
            "NYSE": ("NYSE", "XNYS"),
            "XNAS": ("XNAS", "NASDAQ"),
            "NASDAQ": ("NASDAQ", "XNAS"),
        }
        return alias_map.get(self.calendar_name, (self.calendar_name,))

    def _normalize_calendar_name(self, calendar_name: str) -> str:
        value = calendar_name.strip().upper()
        if not value:
            raise ValueError("calendar_name must be a non-empty string.")
        return value


class XNYSCalendar(MarketCalendar):
    """Backward-compatible market calendar helper defaulting to ``"XNYS"``."""

    def __init__(self, calendar_name: str = "XNYS") -> None:
        super().__init__(calendar_name=calendar_name)
