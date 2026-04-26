"""
Portfolio Metrics
-----------------
Summary statistics over the per-date portfolio returns DataFrame
produced by :func:`run_backtest`.

Functions
---------
compute_portfolio_metrics
    Return a dict of annualized return / vol / Sharpe / max drawdown /
    hit rate / turnover / mean leverage / mean effective N.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["compute_portfolio_metrics"]


def compute_portfolio_metrics(
    *,
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Compute summary statistics for a completed backtest.

    Parameters
    ----------
    returns : pandas.DataFrame
        Rows: one per rebalance date. Columns include ``date``,
        ``portfolio_return``, ``n_assets_held``, ``leverage``, ``hhi``,
        ``effective_n``.
    weights : pandas.DataFrame
        Long-format weights (one row per (date, asset)). Used for
        turnover computation.
    periods_per_year : int, default=252
        Annualization factor. 252 for daily, 52 for weekly, 12 for monthly.

    Returns
    -------
    dict
        Keys: ``annual_return``, ``annual_vol``, ``sharpe``,
        ``max_drawdown``, ``hit_rate``, ``mean_daily_return``,
        ``std_daily_return``, ``mean_turnover``, ``mean_leverage``,
        ``mean_effective_n``, ``n_days``.
    """

    if returns.empty:
        return {
            "annual_return": float("nan"),
            "annual_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "hit_rate": float("nan"),
            "mean_daily_return": float("nan"),
            "std_daily_return": float("nan"),
            "mean_turnover": float("nan"),
            "mean_leverage": float("nan"),
            "mean_effective_n": float("nan"),
            "n_days": 0,
        }

    r = returns["portfolio_return"].dropna()
    n_days = int(len(r))

    mean_daily = float(r.mean())
    std_daily = float(r.std(ddof=0))
    annual_return = float((1.0 + r).prod() ** (periods_per_year / max(1, n_days)) - 1.0)
    annual_vol = std_daily * float(np.sqrt(periods_per_year))
    sharpe = (
        float(mean_daily / std_daily * np.sqrt(periods_per_year))
        if std_daily > 0.0
        else float("nan")
    )
    cumulative = (1.0 + r).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1.0
    max_drawdown = float(drawdown.min())
    hit_rate = float((r > 0.0).mean())

    mean_leverage = float(returns["leverage"].mean()) if "leverage" in returns else float("nan")
    mean_effective_n = (
        float(returns["effective_n"].mean()) if "effective_n" in returns else float("nan")
    )

    turnover = _compute_turnover(weights=weights)

    return {
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "mean_daily_return": mean_daily,
        "std_daily_return": std_daily,
        "mean_turnover": turnover,
        "mean_leverage": mean_leverage,
        "mean_effective_n": mean_effective_n,
        "n_days": n_days,
    }


def _compute_turnover(*, weights: pd.DataFrame) -> float:
    """
    Compute the mean per-rebalance turnover:

        turnover_t = sum_i |w_{i,t} - w_{i,t-1}|
        mean_turnover = mean(turnover_t) over t > 0

    Assets missing in one period have their weight treated as 0 in that
    period (they were either just added or just dropped).
    """

    if weights.empty:
        return float("nan")
    wide = weights.pivot_table(
        index="date", columns="asset_id", values="weight", aggfunc="first"
    ).sort_index()
    if wide.shape[0] < 2:
        return float("nan")
    diffs = wide.fillna(0.0).diff().abs().dropna(how="all")
    return float(diffs.sum(axis=1).mean())
