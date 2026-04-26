"""
Portfolio Run Orchestrator
--------------------------
End-to-end entry point for a portfolio-construction run:

    1. Load predictions.csv + realized-returns CSV
    2. Build alpha / covariance / optimizer components from config
    3. Run the sequential backtest (alpha → Σ → w → apply → record)
    4. Compute summary metrics
    5. Persist manifest + weights.csv + returns.csv + metrics.json/md

Functions
---------
run_portfolio_construction
    Top-level entry point called by ``scripts/construct_portfolio.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from density_model.portfolio.alpha import build_alpha_builder
from density_model.portfolio.backtest import BacktestOutput, run_backtest
from density_model.portfolio.config import PanelPortfolioConfig
from density_model.portfolio.metrics import compute_portfolio_metrics
from density_model.portfolio.optimizer import build_optimizer
from density_model.portfolio.risk_model import build_covariance_estimator

__all__ = ["run_portfolio_construction"]

logger = logging.getLogger(__name__)


def run_portfolio_construction(cfg: PanelPortfolioConfig) -> Path:
    """Execute one portfolio-construction run end-to-end."""

    logger.info("Loading predictions + realized returns")
    predictions = _load_predictions(cfg)
    realized_wide = _load_realized_returns(cfg)

    alpha_builder = build_alpha_builder(cfg.alpha)
    covariance_estimator = build_covariance_estimator(cfg.risk_model)
    optimizer = build_optimizer(cfg.optimizer)

    logger.info(
        "Running backtest: alpha=%s  risk_model=%s(window=%d)  optimizer=%s(λ=%.4g, long_only=%s)",
        cfg.alpha.type,
        cfg.risk_model.type,
        cfg.risk_model.window_size,
        cfg.optimizer.type,
        cfg.optimizer.risk_aversion,
        cfg.optimizer.long_only,
    )
    output = run_backtest(
        cfg=cfg,
        alpha_builder=alpha_builder,
        covariance_estimator=covariance_estimator,
        optimizer=optimizer,
        predictions=predictions,
        realized_wide=realized_wide,
    )

    metrics = compute_portfolio_metrics(
        returns=output.returns,
        weights=output.weights,
        periods_per_year=cfg.backtest.periods_per_year,
    )

    output_dir = cfg.output_path().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "weights.csv"
    returns_path = output_dir / "returns.csv"
    metrics_json_path = output_dir / "metrics.json"
    metrics_md_path = output_dir / "metrics.md"
    manifest_path = output_dir / "portfolio_manifest.json"

    output.weights.to_csv(weights_path, index=False)
    output.returns.to_csv(returns_path, index=False)

    manifest_payload: dict[str, Any] = {
        "run_name": cfg.run_name,
        "predictions_csv": str(cfg.predictions_path().resolve()),
        "realized_returns_csv": str(cfg.realized_returns_path().resolve()),
        "alpha": cfg.alpha.model_dump(),
        "risk_model": cfg.risk_model.model_dump(),
        "optimizer": cfg.optimizer.model_dump(),
        "backtest": cfg.backtest.model_dump(),
        "summary": output.summary,
        "metrics": metrics,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, default=str), encoding="utf-8"
    )
    metrics_json_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    metrics_md_path.write_text(_format_metrics_markdown(metrics, output.summary), encoding="utf-8")

    _print_report(
        cfg=cfg, metrics=metrics, summary=output.summary, output_dir=output_dir
    )
    logger.info("Wrote portfolio artifacts to %s", output_dir)
    return output_dir


def _load_predictions(cfg: PanelPortfolioConfig) -> pd.DataFrame:
    frame = pd.read_csv(cfg.predictions_path())
    required = {
        cfg.id_column,
        cfg.forecast_date_column,
        cfg.mu_column,
        cfg.sigma_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Predictions CSV {cfg.predictions_csv!r} is missing required "
            f"columns: {', '.join(missing)}"
        )
    frame = frame.loc[
        :,
        [cfg.id_column, cfg.forecast_date_column, cfg.mu_column, cfg.sigma_column],
    ].copy()
    frame = frame.rename(
        columns={
            cfg.id_column: "asset_id",
            cfg.forecast_date_column: "forecast_date",
            cfg.mu_column: "mu",
            cfg.sigma_column: "sigma",
        }
    )
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    # Average duplicate (asset_id, forecast_date) rows defensively — some
    # predict paths emit identical rows for t+1 via different rolling
    # windows; taking the first is a no-op in the happy case.
    frame = frame.groupby(["forecast_date", "asset_id"], as_index=True).agg(
        {"mu": "first", "sigma": "first"}
    )
    return frame.sort_index()


def _load_realized_returns(cfg: PanelPortfolioConfig) -> pd.DataFrame:
    frame = pd.read_csv(cfg.realized_returns_path())
    required = {cfg.date_column, cfg.id_column, cfg.return_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Realized-returns CSV {cfg.realized_returns_csv!r} is missing "
            f"required columns: {', '.join(missing)}"
        )
    frame = frame.loc[
        :, [cfg.id_column, cfg.date_column, cfg.return_column]
    ].copy()
    frame[cfg.date_column] = pd.to_datetime(frame[cfg.date_column])
    wide = frame.pivot_table(
        index=cfg.date_column,
        columns=cfg.id_column,
        values=cfg.return_column,
        aggfunc="first",
    ).sort_index()
    return wide


def _format_metrics_markdown(metrics: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = ["# Portfolio Backtest Summary\n", "## Performance\n"]
    lines.extend(
        [
            f"- **Annual return** : {_fmt(metrics['annual_return'], '.4%')}",
            f"- **Annual vol**    : {_fmt(metrics['annual_vol'], '.4%')}",
            f"- **Sharpe**        : {_fmt(metrics['sharpe'], '.3f')}",
            f"- **Max drawdown**  : {_fmt(metrics['max_drawdown'], '.4%')}",
            f"- **Hit rate**      : {_fmt(metrics['hit_rate'], '.4f')}",
            "",
            "## Portfolio composition",
            f"- **Mean turnover**  : {_fmt(metrics['mean_turnover'], '.4f')}",
            f"- **Mean leverage**  : {_fmt(metrics['mean_leverage'], '.3f')}",
            f"- **Mean effective N**: {_fmt(metrics['mean_effective_n'], '.2f')}",
            "",
            "## Window",
            f"- Rebalance dates : {summary['n_rebalance_dates']}",
            f"- Start / end     : {summary.get('start_date')} / {summary.get('end_date')}",
            f"- Skipped (burn-in): {summary.get('skipped_burn_in_dates', 0)}",
            f"- Skipped (missing-asset): {summary.get('skipped_missing_asset_dates', 0)}",
            f"- Skipped (thin universe): {summary.get('skipped_insufficient_universe_dates', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"


def _print_report(
    *,
    cfg: PanelPortfolioConfig,
    metrics: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    separator = "─" * 72
    print(separator)
    print(f"Portfolio backtest — {cfg.run_name or output_dir.name}")
    print(separator)
    print(f"  predictions    : {cfg.predictions_csv}")
    print(f"  realized       : {cfg.realized_returns_csv}")
    print(
        f"  alpha / risk   : {cfg.alpha.type}  |  {cfg.risk_model.type} "
        f"window={cfg.risk_model.window_size}  ridge={cfg.risk_model.ridge:g}"
    )
    print(
        f"  optimizer      : {cfg.optimizer.type}  λ={cfg.optimizer.risk_aversion:g}  "
        f"long_only={cfg.optimizer.long_only}  full_investment={cfg.optimizer.full_investment}"
    )
    print(
        f"  window         : {summary.get('start_date')} → {summary.get('end_date')} "
        f"({summary.get('n_rebalance_dates')} rebalance dates; "
        f"{summary.get('skipped_burn_in_dates', 0)} burn-in, "
        f"{summary.get('skipped_missing_asset_dates', 0)} missing-asset, "
        f"{summary.get('skipped_insufficient_universe_dates', 0)} thin-universe skipped)"
    )
    print(separator)
    print("Performance")
    print(f"  annual_return   = {_fmt(metrics['annual_return'], '+.4%')}")
    print(f"  annual_vol      = {_fmt(metrics['annual_vol'], '.4%')}")
    print(f"  sharpe          = {_fmt(metrics['sharpe'], '+.3f')}")
    print(f"  max_drawdown    = {_fmt(metrics['max_drawdown'], '.4%')}")
    print(f"  hit_rate        = {_fmt(metrics['hit_rate'], '.4f')}")
    print(separator)
    print("Portfolio composition")
    print(f"  mean_turnover   = {_fmt(metrics['mean_turnover'], '.4f')}")
    print(f"  mean_leverage   = {_fmt(metrics['mean_leverage'], '.3f')}")
    print(f"  mean_effective_n= {_fmt(metrics['mean_effective_n'], '.2f')}")
    print(separator)
    print(f"Wrote weights.csv + returns.csv + metrics.{{json,md}} to {output_dir}")


def _fmt(value: float, spec: str) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (value != value):  # NaN check
        return "—"
    return f"{value:{spec}}"
