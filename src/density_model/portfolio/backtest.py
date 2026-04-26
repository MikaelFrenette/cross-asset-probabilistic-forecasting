"""
Backtest Loop
-------------
Strict-no-lookahead sequential backtest:

    for each forecast date t+1 in the backtest window:
        1. read (μ, σ) for t+1 from predictions.csv
        2. drop assets with missing μ/σ (per ``missing_asset_policy``)
        3. estimate Σ_t using realized returns strictly before t+1
        4. α_t = zscore(μ/σ) across the valid cross-section
        5. w_t = argmax_w α_t'w − (λ/2) w'Σ_t w   subject to 1'w = 1
        6. realize r_{t+1}; portfolio return = w_t · r_{t+1}
        7. record weights + realized portfolio return + bookkeeping

Functions
---------
run_backtest
    Run the sequential loop and return per-date weights + portfolio
    returns as two DataFrames (+ summary metadata).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from density_model.portfolio.alpha.base import BaseAlphaBuilder
from density_model.portfolio.config import PanelPortfolioConfig
from density_model.portfolio.optimizer.base import BaseOptimizer
from density_model.portfolio.risk_model.base import BaseCovarianceEstimator

__all__ = ["BacktestOutput", "run_backtest"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestOutput:
    """Frozen bundle of backtest artifacts."""

    weights: pd.DataFrame
    returns: pd.DataFrame
    summary: dict[str, Any]


def run_backtest(
    *,
    cfg: PanelPortfolioConfig,
    alpha_builder: BaseAlphaBuilder,
    covariance_estimator: BaseCovarianceEstimator,
    optimizer: BaseOptimizer,
    predictions: pd.DataFrame,
    realized_wide: pd.DataFrame,
) -> BacktestOutput:
    """
    Execute the sequential backtest.

    Parameters
    ----------
    cfg : PanelPortfolioConfig
        Validated top-level config (provides window, policies, etc.).
    alpha_builder, covariance_estimator, optimizer : callable bundles
        Built by the respective ``build_*`` dispatchers.
    predictions : pandas.DataFrame
        Long-format predictions with columns (``asset_id``,
        ``forecast_date``, ``mu``, ``sigma``). Index is a
        ``MultiIndex(forecast_date, asset_id)`` sorted in that order.
    realized_wide : pandas.DataFrame
        Realized returns reshaped to wide format: rows = dates, columns
        = asset_ids. Index is a monotonically-increasing DatetimeIndex.

    Returns
    -------
    BacktestOutput
        ``weights`` (long format), ``returns`` (one row per backtest date),
        and a ``summary`` dict with counts + window metadata.
    """

    burn_in = cfg.backtest.burn_in_days or cfg.risk_model.window_size
    missing_policy = cfg.backtest.missing_asset_policy

    forecast_dates = predictions.index.get_level_values("forecast_date").unique().sort_values()
    realized_dates = realized_wide.index.sort_values()

    start_bound = _parse_date(cfg.backtest.start_date) or forecast_dates.min()
    end_bound = _parse_date(cfg.backtest.end_date) or forecast_dates.max()

    weights_rows: list[dict[str, Any]] = []
    returns_rows: list[dict[str, Any]] = []
    skipped_missing_dates = 0
    skipped_burn_in = 0
    skipped_insufficient_universe = 0

    for forecast_date in forecast_dates:
        if forecast_date < start_bound or forecast_date > end_bound:
            continue

        cross_section = predictions.xs(forecast_date, level="forecast_date")
        valid = cross_section.dropna(subset=["mu", "sigma"])

        if missing_policy == "drop_date" and len(valid) < len(cross_section):
            skipped_missing_dates += 1
            continue

        if len(valid) < 2:
            skipped_insufficient_universe += 1
            continue

        history = realized_wide.loc[realized_wide.index < forecast_date, valid.index]
        if history.empty:
            skipped_burn_in += 1
            continue

        # ``as_of`` is the last observable realized date strictly before the
        # forecast date — the no-lookahead boundary.
        as_of_candidates = history.index
        if len(as_of_candidates) == 0:
            skipped_burn_in += 1
            continue
        as_of = as_of_candidates.max()

        covariance = covariance_estimator.estimate(history, as_of=as_of)
        if covariance is None:
            skipped_burn_in += 1
            continue
        if len(covariance.index) < burn_in and len(history) < burn_in:
            # belt-and-braces guard; estimator should already have returned None
            skipped_burn_in += 1
            continue

        alpha = alpha_builder.compute(mu=valid["mu"], sigma=valid["sigma"])
        alpha = alpha.dropna()
        common = alpha.index.intersection(covariance.index)
        alpha = alpha.loc[common]
        if len(alpha) < 2:
            skipped_insufficient_universe += 1
            continue
        covariance = covariance.loc[common, common]

        weights = optimizer.optimize(alpha=alpha, covariance=covariance)

        realized_today = realized_wide.loc[forecast_date, weights.index]
        realized_today = realized_today.dropna()
        aligned_weights = weights.reindex(realized_today.index)
        if aligned_weights.empty:
            skipped_insufficient_universe += 1
            continue
        portfolio_return = float((aligned_weights * realized_today).sum())
        leverage = float(aligned_weights.abs().sum())
        hhi = float((aligned_weights ** 2).sum())
        effective_n = float(1.0 / hhi) if hhi > 0.0 else float("nan")

        for asset, weight in aligned_weights.items():
            weights_rows.append(
                {
                    "date": forecast_date,
                    "asset_id": asset,
                    "weight": float(weight),
                    "alpha": float(alpha.get(asset, float("nan"))),
                    "realized_return": float(realized_today[asset]),
                }
            )
        returns_rows.append(
            {
                "date": forecast_date,
                "portfolio_return": portfolio_return,
                "n_assets_held": int(len(aligned_weights)),
                "leverage": leverage,
                "hhi": hhi,
                "effective_n": effective_n,
            }
        )

    weights_frame = pd.DataFrame(
        weights_rows,
        columns=["date", "asset_id", "weight", "alpha", "realized_return"],
    )
    returns_frame = pd.DataFrame(
        returns_rows,
        columns=[
            "date",
            "portfolio_return",
            "n_assets_held",
            "leverage",
            "hhi",
            "effective_n",
        ],
    )

    summary = {
        "n_rebalance_dates": int(len(returns_frame)),
        "n_unique_assets": int(weights_frame["asset_id"].nunique()) if not weights_frame.empty else 0,
        "start_date": returns_frame["date"].min().isoformat() if not returns_frame.empty else None,
        "end_date": returns_frame["date"].max().isoformat() if not returns_frame.empty else None,
        "skipped_burn_in_dates": skipped_burn_in,
        "skipped_missing_asset_dates": skipped_missing_dates,
        "skipped_insufficient_universe_dates": skipped_insufficient_universe,
        "burn_in_days": burn_in,
        "missing_asset_policy": missing_policy,
    }
    logger.info(
        "Backtest complete — %d rebalance dates (%s → %s); skipped %d (burn-in), "
        "%d (missing-asset-date), %d (insufficient universe)",
        summary["n_rebalance_dates"],
        summary["start_date"],
        summary["end_date"],
        skipped_burn_in,
        skipped_missing_dates,
        skipped_insufficient_universe,
    )
    return BacktestOutput(
        weights=weights_frame, returns=returns_frame, summary=summary
    )


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    return pd.Timestamp(value)
