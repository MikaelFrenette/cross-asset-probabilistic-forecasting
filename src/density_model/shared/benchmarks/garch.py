"""
GARCH(1,1) Benchmark
--------------------
Per-asset Constant-mean + GARCH(1,1) via the ``arch`` package. Returns are
scaled by 100 internally for numerical stability during fit, then
unscaled on output so ``mu`` and ``sigma`` are reported in the same
return space as ``predictions.csv``.

At prediction time the variance is rolled forward one step at a time
using the fitted ``(omega, alpha, beta)`` and the realized residuals
observed during the test window — a proper one-step-ahead
out-of-sample rollout.

Emits full density forecasts (``mu`` + ``sigma`` per (asset, date)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density_model.shared.benchmarks.base import BaseBenchmark

__all__ = ["GarchBenchmark"]

_RETURN_SCALE = 100.0   # percent scale used internally by arch for stability


class GarchBenchmark(BaseBenchmark):
    """Per-asset Constant-mean + GARCH(1,1) one-step-ahead density forecaster."""

    def __init__(self) -> None:
        self.params_: dict[str, dict[str, float]] = {}
        self.last_variance_: dict[str, float] = {}

    def fit(
        self,
        train_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> None:
        try:
            from arch import arch_model
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "The `arch` package is required for the GARCH(1,1) benchmark. "
                "Install it via `pip install arch`."
            ) from error

        for asset, group in train_panel.groupby(id_column, sort=False):
            series = (
                group.sort_values(date_column)[target_column]
                .dropna()
                .to_numpy(dtype=float)
            )
            if series.size < 60:
                continue  # too little history to fit GARCH reliably
            scaled = series * _RETURN_SCALE
            am = arch_model(
                scaled,
                mean="Constant",
                vol="Garch",
                p=1,
                q=1,
                rescale=False,
            )
            try:
                res = am.fit(disp="off", show_warning=False)
            except Exception as error:  # pragma: no cover
                # GARCH fit can fail on degenerate series; fall back to a flat model.
                continue
            params = res.params.to_dict()
            self.params_[str(asset)] = {
                "mu": float(params.get("mu", 0.0)),
                "omega": float(params["omega"]),
                "alpha": float(params.get("alpha[1]", 0.0)),
                "beta": float(params.get("beta[1]", 0.0)),
            }
            # Terminal in-sample variance carries forward into the first test step.
            cond_vol = res.conditional_volatility
            last_vol = cond_vol.iloc[-1] if hasattr(cond_vol, "iloc") else cond_vol[-1]
            self.last_variance_[str(asset)] = float(last_vol) ** 2

    def predict(
        self,
        predict_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for asset, group in predict_panel.groupby(id_column, sort=False):
            key = str(asset)
            if key not in self.params_:
                continue
            params = self.params_[key]
            mu_scaled = params["mu"]
            omega = params["omega"]
            alpha = params["alpha"]
            beta = params["beta"]

            var_t = self.last_variance_.get(key, float("nan"))
            group_sorted = group.sort_values(date_column).reset_index(drop=True)
            realized = group_sorted[target_column].to_numpy(dtype=float)
            dates = group_sorted[date_column].to_list()
            for i, date in enumerate(dates):
                if np.isnan(var_t) or var_t <= 0.0:
                    sigma_raw = float("nan")
                else:
                    sigma_scaled = var_t ** 0.5
                    sigma_raw = sigma_scaled / _RETURN_SCALE
                rows.append(
                    {
                        "forecast_start_date": date,
                        "forecast_end_date": date,
                        "asset_id": asset,
                        "mu": mu_scaled / _RETURN_SCALE,
                        "sigma": sigma_raw,
                    }
                )
                # Update variance using realized residual (in scaled space).
                if not np.isnan(realized[i]):
                    resid = realized[i] * _RETURN_SCALE - mu_scaled
                    var_t = omega + alpha * resid * resid + beta * var_t
        return pd.DataFrame(rows).loc[
            :, ["forecast_start_date", "forecast_end_date", "asset_id", "mu", "sigma"]
        ]
