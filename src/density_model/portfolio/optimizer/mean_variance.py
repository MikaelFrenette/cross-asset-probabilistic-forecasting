"""
Mean-Variance Optimizer
-----------------------
Closed-form solution to

    max_{w} α'w − (λ/2) w'Σw        subject to     1'w = 1                 (*)

via the Lagrangian:

    L(w, ν) = α'w − (λ/2) w'Σw − ν (1'w − 1)

Stationarity gives

    α − λΣw − ν 1 = 0      ⇒     w = (1/λ) Σ⁻¹ (α − ν 1)

and the constraint gives

    ν = (1' Σ⁻¹ α − λ) / (1' Σ⁻¹ 1)

so the unique solution is

    w = (1/λ) Σ⁻¹ (α − ν 1).

When ``long_only: true`` the closed form no longer applies (the
non-negativity constraint is active). The optimizer falls back to
``scipy.optimize.minimize`` with SLSQP, initialized at the equal-weight
portfolio.

When ``full_investment: false`` the ``1'w = 1`` constraint is relaxed
and the closed-form reduces to ``w = (1/λ) Σ⁻¹ α`` (ν = 0); under
long-only without full-investment, SLSQP still applies with only the
bounds ``w ≥ 0`` active.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density_model.portfolio.optimizer.base import BaseOptimizer

__all__ = ["MeanVarianceOptimizer"]


class MeanVarianceOptimizer(BaseOptimizer):
    """``α − (λ/2) w'Σw`` with optional full-investment and long-only knobs."""

    def __init__(
        self,
        *,
        risk_aversion: float,
        long_only: bool = False,
        full_investment: bool = True,
    ) -> None:
        if risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be > 0.")
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.full_investment = full_investment

    def optimize(
        self,
        *,
        alpha: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        assets = alpha.index
        if len(assets) < 1:
            return pd.Series(dtype=float)
        cov = covariance.loc[assets, assets].to_numpy(dtype=float)
        a = alpha.to_numpy(dtype=float)

        if self.long_only:
            return self._solve_long_only(assets=assets, alpha=a, covariance=cov)
        return self._solve_closed_form(assets=assets, alpha=a, covariance=cov)

    def _solve_closed_form(
        self, *, assets: pd.Index, alpha: np.ndarray, covariance: np.ndarray
    ) -> pd.Series:
        inv_cov_alpha = np.linalg.solve(covariance, alpha)
        if self.full_investment:
            ones = np.ones(len(assets))
            inv_cov_ones = np.linalg.solve(covariance, ones)
            ones_sigma_ones = float(ones @ inv_cov_ones)
            if ones_sigma_ones <= 0.0:
                raise ValueError(
                    "Full-investment closed form requires 1'Σ⁻¹1 > 0; got "
                    f"{ones_sigma_ones}."
                )
            ones_sigma_alpha = float(ones @ inv_cov_alpha)
            nu = (ones_sigma_alpha - self.risk_aversion) / ones_sigma_ones
            w = (1.0 / self.risk_aversion) * (inv_cov_alpha - nu * inv_cov_ones)
        else:
            w = inv_cov_alpha / self.risk_aversion
        return pd.Series(w, index=assets, name="weight")

    def _solve_long_only(
        self, *, assets: pd.Index, alpha: np.ndarray, covariance: np.ndarray
    ) -> pd.Series:
        try:
            from scipy.optimize import minimize
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "scipy is required for long_only=true. "
                "Install scipy or set long_only: false."
            ) from error

        n = len(assets)
        x0 = np.full(n, 1.0 / n)
        bounds = [(0.0, None)] * n
        constraints = []
        if self.full_investment:
            constraints.append(
                {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
            )

        def negative_objective(weights: np.ndarray) -> float:
            return -(
                float(alpha @ weights)
                - 0.5 * self.risk_aversion * float(weights @ covariance @ weights)
            )

        def negative_gradient(weights: np.ndarray) -> np.ndarray:
            return -(alpha - self.risk_aversion * (covariance @ weights))

        result = minimize(
            negative_objective,
            x0,
            jac=negative_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(
                f"Long-only MVO optimizer failed to converge: {result.message!r}"
            )
        weights = np.clip(result.x, 0.0, None)
        if self.full_investment and weights.sum() > 0.0:
            weights = weights / weights.sum()
        return pd.Series(weights, index=assets, name="weight")
