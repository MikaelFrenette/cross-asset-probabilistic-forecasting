"""
Portfolio Cumulative-Return Plot
--------------------------------
Equity-curve plot comparing the backtested portfolio against one or more
benchmark assets pulled from the same realized-returns CSV used during
the backtest. Both curves are normalized to **1.0 at the trading day
before the first rebalance** — every subsequent point compounds from
there.

Functions
---------
plot_portfolio_vs_benchmarks
    Render and save a single PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["plot_portfolio_vs_benchmarks"]

logger = logging.getLogger(__name__)

_BENCHMARK_PALETTE = ("#fb7185", "#f59e0b", "#a78bfa", "#67e8f9", "#86efac", "#fde68a")


def plot_portfolio_vs_benchmarks(
    *,
    portfolio_returns_csv: Path,
    realized_returns_csv: Path,
    date_column: str = "date",
    id_column: str = "asset_id",
    return_column: str = "return",
    benchmark_asset_ids: Sequence[str] = ("SPY",),
    output_path: Path,
    title_suffix: str | None = None,
) -> Path:
    """
    Render the cumulative-return plot.

    Parameters
    ----------
    portfolio_returns_csv : pathlib.Path
        ``returns.csv`` produced by :func:`run_portfolio_construction`
        (columns include ``date`` + ``portfolio_return``).
    realized_returns_csv : pathlib.Path
        Same realized-returns CSV used by the backtest. Must contain
        rows for every ``benchmark_asset_ids`` ticker.
    date_column, id_column, return_column : str
        Column names in ``realized_returns_csv``.
    benchmark_asset_ids : sequence of str, default ("SPY",)
        Tickers to overlay on the portfolio curve.
    output_path : pathlib.Path
        PNG output path. Parent directory is created if needed.
    title_suffix : str, optional
        Extra text appended to the figure subtitle (after the date range
        and final-value summary).
    """

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for portfolio plots."
        ) from error

    portfolio_returns = _load_portfolio_returns(portfolio_returns_csv)
    benchmarks = _load_benchmark_returns(
        realized_returns_csv,
        date_column=date_column,
        id_column=id_column,
        return_column=return_column,
        asset_ids=tuple(benchmark_asset_ids),
    )

    # Trading-day grid common to portfolio + every benchmark.
    common_dates = portfolio_returns.index
    for series in benchmarks.values():
        common_dates = common_dates.intersection(series.index)
    common_dates = common_dates.sort_values()
    if len(common_dates) == 0:
        raise ValueError(
            "No overlapping dates between the portfolio returns and the requested "
            "benchmark assets."
        )

    portfolio_index = _cumulative_index(portfolio_returns.loc[common_dates])
    benchmark_indices = {
        asset_id: _cumulative_index(series.loc[common_dates])
        for asset_id, series in benchmarks.items()
    }

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#42526b",
            "axes.labelcolor": "#d9e2ec",
            "xtick.color": "#cbd5e1",
            "ytick.color": "#cbd5e1",
            "text.color": "#e5edf5",
            "figure.facecolor": "#08111f",
            "axes.facecolor": "#0b1728",
            "savefig.facecolor": "#08111f",
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        portfolio_index.index,
        portfolio_index.values,
        color="#22d3ee",
        linewidth=2.0,
        label="Portfolio",
    )
    for offset, (asset_id, idx) in enumerate(benchmark_indices.items()):
        color = _BENCHMARK_PALETTE[offset % len(_BENCHMARK_PALETTE)]
        ax.plot(
            idx.index,
            idx.values,
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label=str(asset_id),
        )

    ax.axhline(1.0, color="#334155", linewidth=0.7, linestyle="--", alpha=0.7)

    title_dates = f"{pd.Timestamp(common_dates[0]).date()} – {pd.Timestamp(common_dates[-1]).date()}"
    final_strs = [f"Portfolio {portfolio_index.iloc[-1]:.2f}×"] + [
        f"{asset_id} {idx.iloc[-1]:.2f}×"
        for asset_id, idx in benchmark_indices.items()
    ]
    subtitle = f"{title_dates}   |   {'   ·   '.join(final_strs)}"
    if title_suffix:
        subtitle = f"{subtitle}   |   {title_suffix}"

    fig.suptitle(
        "Cumulative Returns — Portfolio vs Benchmarks",
        x=0.06,
        y=0.96,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#f8fafc",
    )
    fig.text(
        0.06,
        0.93,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color="#7dd3fc",
    )

    ax.set_ylabel("Cumulative return (×, log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", color="#223449", alpha=0.7, linewidth=0.7)
    ax.grid(axis="x", color="#162335", alpha=0.45, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#42526b")
    ax.spines["bottom"].set_color("#42526b")

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=11, labelcolor="#e5edf5")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    logger.info("Wrote cumulative-return plot to %s", output_path)
    return output_path


def _load_portfolio_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "portfolio_return" not in frame.columns:
        raise ValueError(
            f"Portfolio returns CSV {path!s} is missing 'date' or 'portfolio_return' columns."
        )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date")["portfolio_return"].sort_index()


def _load_benchmark_returns(
    path: Path,
    *,
    date_column: str,
    id_column: str,
    return_column: str,
    asset_ids: tuple[str, ...],
) -> dict[str, pd.Series]:
    frame = pd.read_csv(path)
    required = {date_column, id_column, return_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Realized-returns CSV {path!s} is missing required columns: {', '.join(missing)}"
        )
    frame[date_column] = pd.to_datetime(frame[date_column])
    benchmarks: dict[str, pd.Series] = {}
    for asset_id in asset_ids:
        sub = frame.loc[frame[id_column] == asset_id, [date_column, return_column]]
        if sub.empty:
            raise ValueError(
                f"Benchmark asset {asset_id!r} not found in realized-returns CSV."
            )
        series = sub.set_index(date_column)[return_column].sort_index()
        benchmarks[asset_id] = series
    return benchmarks


def _cumulative_index(returns: pd.Series) -> pd.Series:
    """Return cumulative ``(1+r).cumprod()`` prepended with 1.0 one day earlier."""

    if returns.empty:
        return returns
    returns_clean = returns.dropna()
    if returns_clean.empty:
        return returns_clean
    cumulative = (1.0 + returns_clean).cumprod()
    first_date = pd.Timestamp(returns_clean.index[0])
    seed_date = first_date - pd.Timedelta(days=1)
    seed = pd.Series([1.0], index=[seed_date])
    return pd.concat([seed, cumulative]).sort_index()
