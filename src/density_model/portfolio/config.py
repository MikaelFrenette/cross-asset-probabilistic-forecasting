"""
Panel Portfolio Configuration
-----------------------------
Top-level Pydantic config for a portfolio-construction run. Sibling to
:class:`PanelPredictConfig` / :class:`PanelBenchmarkConfig` — stands
alone, reads a ``predictions.csv`` (from any model or benchmark) plus a
realized-returns CSV, and drives a mean-variance backtest.

Every decision in the pipeline (alpha style, covariance estimator, risk
aversion, long-only, burn-in, missing-asset policy, rebalance rules) is
a knob under one of the nested sections so later iterations can swap
variants without touching code.

Classes
-------
PanelAlphaConfig
    Alpha-construction knobs (type dispatch).
PanelRiskModelConfig
    Covariance estimator knobs (type + window size + ridge regularizer).
PanelOptimizerConfig
    Optimizer knobs (risk aversion λ, long-only, full-investment).
PanelBacktestConfig
    Window + burn-in + missing-asset policy knobs.
PanelPortfolioConfig
    Top-level container.

Functions
---------
load_panel_portfolio_config
    Load + validate a YAML config.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from density_model.shared.config.panel_schema import PanelLoggingConfig, _validate_non_empty

__all__ = [
    "PanelAlphaConfig",
    "PanelBacktestConfig",
    "PanelOptimizerConfig",
    "PanelPortfolioConfig",
    "PanelRiskModelConfig",
    "load_panel_portfolio_config",
]


class PanelAlphaConfig(BaseModel):
    """Alpha construction from (μ, σ)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(default="mu_sigma_zscore", pattern="^(mu_sigma_zscore)$")


class PanelRiskModelConfig(BaseModel):
    """Covariance estimator knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # "rolling" only for v1; expanding / ewma / shrinkage planned as extension sites.
    type: str = Field(default="rolling", pattern="^(rolling)$")
    window_size: int = Field(default=126, ge=2)
    ridge: float = Field(default=1e-4, ge=0.0)
    ridge_mode: str = Field(default="relative", pattern="^(relative|absolute)$")


class PanelOptimizerConfig(BaseModel):
    """Mean-variance optimizer knobs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(default="mean_variance", pattern="^(mean_variance)$")
    risk_aversion: float = Field(default=1.0, gt=0.0)
    long_only: bool = False
    full_investment: bool = True


class PanelBacktestConfig(BaseModel):
    """Backtest window + missing-data policies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # ISO-format dates; None → use predictions earliest / latest with one-day trim
    start_date: str | None = None
    end_date: str | None = None
    # None → default to risk_model.window_size (set at build time)
    burn_in_days: int | None = None
    missing_asset_policy: str = Field(
        default="drop_asset", pattern="^(drop_asset|drop_date)$"
    )
    periods_per_year: int = Field(default=252, ge=1)


class PanelPortfolioConfig(BaseModel):
    """Top-level config for a portfolio-construction run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Inputs
    predictions_csv: str
    realized_returns_csv: str
    date_column: str = "date"
    id_column: str = "asset_id"
    return_column: str = "return"

    # Prediction-CSV column names (match scripts/predict.py output by default)
    forecast_date_column: str = "forecast_start_date"
    mu_column: str = "mu"
    sigma_column: str = "sigma"

    # Pipeline sections
    alpha: PanelAlphaConfig = Field(default_factory=PanelAlphaConfig)
    risk_model: PanelRiskModelConfig = Field(default_factory=PanelRiskModelConfig)
    optimizer: PanelOptimizerConfig = Field(default_factory=PanelOptimizerConfig)
    backtest: PanelBacktestConfig = Field(default_factory=PanelBacktestConfig)

    # Output
    output_dir: str
    run_name: str | None = None

    # Logging
    logging: PanelLoggingConfig = Field(default_factory=PanelLoggingConfig)

    @field_validator(
        "predictions_csv",
        "realized_returns_csv",
        "date_column",
        "id_column",
        "return_column",
        "forecast_date_column",
        "mu_column",
        "sigma_column",
        "output_dir",
    )
    @classmethod
    def _non_empty(cls, v: str) -> str:
        return _validate_non_empty(v)

    def predictions_path(self) -> Path:
        return Path(self.predictions_csv)

    def realized_returns_path(self) -> Path:
        return Path(self.realized_returns_csv)

    def output_path(self) -> Path:
        return Path(self.output_dir)


def load_panel_portfolio_config(path: str | Path) -> PanelPortfolioConfig:
    """Load + validate a YAML portfolio config."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Configuration file {path!r} is empty.")
    if not isinstance(payload, dict):
        raise TypeError(
            f"Configuration file {path!r} must contain a top-level mapping."
        )
    return PanelPortfolioConfig.model_validate(payload)
