"""
Gaussian Ensemble Aggregation
-----------------------------
Aggregate per-estimator Gaussian predictions into a single predictive
distribution. Because each estimator emits its own ``(mu, sigma)``, the
ensemble sigma must combine both within-estimator variance (mean of squared
sigmas) and between-estimator variance (variance of means). This is the law
of total variance:

    sigma_ens^2 = E[sigma_k^2] + Var[mu_k]

Using ``"median"`` aggregation replaces the E[mu] term with the median of the
estimator means but keeps the sigma formula unchanged.

Functions
---------
aggregate_gaussian_predictions
    Reduce a stack of ``(mu_k, sigma_k)`` arrays to ``(mu_ens, sigma_ens)``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["aggregate_gaussian_predictions"]


def aggregate_gaussian_predictions(
    *,
    mus: np.ndarray,
    sigmas: np.ndarray,
    mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-estimator Gaussians into a single ``(mu, sigma)`` pair.

    Parameters
    ----------
    mus : numpy.ndarray
        Per-estimator mean predictions shaped ``(K, ...)``.
    sigmas : numpy.ndarray
        Per-estimator standard deviations shaped ``(K, ...)``.
    mode : {"mean", "median"}, default="mean"
        Aggregation used for the predictive mean. ``"mean"`` takes the
        arithmetic mean of ``mu_k``; ``"median"`` takes their median.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Ensemble mean and ensemble standard deviation broadcast against the
        non-estimator axes of the inputs.
    """

    if mus.shape != sigmas.shape:
        raise ValueError("mus and sigmas must share the same shape.")
    if mus.shape[0] == 0:
        raise ValueError("Cannot aggregate an empty ensemble.")

    if mode == "mean":
        mu_ens = mus.mean(axis=0)
    elif mode == "median":
        mu_ens = np.median(mus, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation mode {mode!r}; expected 'mean' or 'median'.")

    within_variance = (sigmas ** 2).mean(axis=0)
    between_variance = mus.var(axis=0, ddof=0)
    sigma_ens = np.sqrt(within_variance + between_variance)
    return mu_ens, sigma_ens
