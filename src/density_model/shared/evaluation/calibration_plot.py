"""
Sigma Calibration Plot
----------------------
Two-panel showcase of the model's *probabilistic* output:

1. **Conditional volatility** — predicted ``sigma`` for one focus asset
   overlaid with rolling-window realized volatility. Demonstrates that the
   model's σ tracks the underlying volatility regime, not just point μ.

2. **Reliability diagram** — empirical coverage of the standardized
   residual ``z = (y - mu) / sigma`` against the nominal Gaussian
   coverage at a panel of symmetric prediction-interval levels. Points
   landing on the 45° identity line indicate a well-calibrated
   Gaussian predictive distribution.

The plot is computed from a ``PanelPredictConfig`` (the same YAML
``scripts/predict.py`` and ``scripts/plot.py`` already consume), so it
works for both single-model and ensemble manifests.

Functions
---------
generate_calibration_plot
    Entry point called by ``scripts/plot_calibration.py``. Renders a
    PNG to ``output_path`` (defaults to ``<predictions-dir>/sigma_calibration.png``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import erfinv as _scipy_erfinv

from density_model.shared.config.panel_schema import PanelPredictConfig
from density_model.shared.ensembling.panel.manifest import load_panel_ensemble_manifest
from density_model.shared.training.panel_manifest import load_panel_manifest

__all__ = ["generate_calibration_plot"]

logger = logging.getLogger(__name__)


DEFAULT_NOMINAL_LEVELS: tuple[float, ...] = (
    0.50,
    0.60,
    0.6827,  # ±1σ
    0.80,
    0.90,
    0.9545,  # ±2σ
    0.99,
)


def generate_calibration_plot(
    cfg: PanelPredictConfig,
    *,
    output_path: Path | None = None,
    focus_asset: str = "SPY",
    realized_vol_window: int = 21,
    nominal_levels: Sequence[float] = DEFAULT_NOMINAL_LEVELS,
) -> Path:
    """
    Render the two-panel σ calibration figure.

    Parameters
    ----------
    cfg : PanelPredictConfig
        Validated panel prediction config — same one that drives
        ``predict.py`` / ``plot.py``.
    output_path : pathlib.Path | None
        Destination PNG. Defaults to ``<predictions-dir>/sigma_calibration.png``.
    focus_asset : str
        Ticker whose σ time series populates the left panel.
    realized_vol_window : int
        Rolling-window size (trading days) used for the realized
        volatility comparison line.
    nominal_levels : Sequence[float]
        Symmetric prediction-interval levels (in (0, 1)) at which to
        evaluate empirical coverage.

    Returns
    -------
    pathlib.Path
        The written PNG path.
    """

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except ImportError as error:
        raise ImportError(
            "matplotlib must be installed to render the calibration plot."
        ) from error

    target_column, model_label = _resolve_manifest_metadata(cfg.manifest_file())
    predictions = _load_predictions(cfg.output_path())
    realized = _load_realized_series(
        csv_path=cfg.data.csv_file(),
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        target_column=target_column,
    )
    joined = predictions.join(realized.rename(columns={target_column: "target"}), how="inner")
    if joined.empty:
        raise ValueError("Calibration plot requires overlapping realized targets for forecast dates.")

    if output_path is None:
        output_path = cfg.output_path().parent / "sigma_calibration.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coverage = _coverage_table(joined=joined, nominal_levels=nominal_levels)

    asset_index = joined.index.get_level_values("asset_id").unique()
    if focus_asset not in asset_index:
        focus_asset = sorted(asset_index)[0]
        logger.warning("Focus asset not found in predictions; falling back to %s", focus_asset)

    focus_frame = joined.xs(focus_asset, level="asset_id").sort_index()
    sigma = focus_frame["sigma"]
    realized_vol = _compute_realized_volatility(
        y=focus_frame["target"], window=realized_vol_window
    )

    _set_dark_style(plt)
    fig, (ax_vol, ax_cal) = plt.subplots(
        ncols=2,
        figsize=(15.5, 5.6),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    fig.subplots_adjust(top=0.84, wspace=0.22)

    fig.suptitle(
        "Probabilistic forecast — σ calibration",
        x=0.06,
        y=0.965,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color="#f8fafc",
    )
    fig.text(
        0.06,
        0.928,
        f"model={model_label}   |   panel coverage measured over "
        f"{len(joined):,} (asset, date) forecasts",
        ha="left",
        va="top",
        fontsize=10,
        color="#7dd3fc",
    )

    _render_volatility_panel(
        ax=ax_vol,
        sigma=sigma,
        realized_vol=realized_vol,
        focus_asset=focus_asset,
        window=realized_vol_window,
        mdates=mdates,
        FormatStrFormatter=FormatStrFormatter,
    )
    _render_reliability_panel(
        ax=ax_cal,
        coverage=coverage,
        FormatStrFormatter=FormatStrFormatter,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote calibration plot to %s", output_path)
    return output_path


def _coverage_table(
    *, joined: pd.DataFrame, nominal_levels: Sequence[float]
) -> pd.DataFrame:
    """Return per-level nominal vs empirical coverage."""

    mu = joined["mu"].to_numpy()
    sigma = joined["sigma"].to_numpy()
    target = joined["target"].to_numpy()
    mask = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(target) & (sigma > 0.0)
    z = (target[mask] - mu[mask]) / sigma[mask]

    nominal_values: list[float] = []
    empirical_values: list[float] = []
    half_widths: list[float] = []
    for nominal in nominal_levels:
        if not 0.0 < nominal < 1.0:
            raise ValueError(f"Nominal coverage level must lie in (0, 1); got {nominal!r}.")
        half_width = _half_width_from_level(nominal)
        empirical = float(np.mean(np.abs(z) <= half_width))
        nominal_values.append(float(nominal))
        empirical_values.append(empirical)
        half_widths.append(half_width)

    return pd.DataFrame(
        {
            "nominal": nominal_values,
            "empirical": empirical_values,
            "half_width": half_widths,
        }
    )


def _half_width_from_level(level: float) -> float:
    """Return the standard-normal half-width for a symmetric coverage ``level``."""

    if not 0.0 < level < 1.0:
        raise ValueError(f"Coverage level must lie in (0, 1); got {level!r}.")
    return float(np.sqrt(2.0) * _scipy_erfinv(level))


def _render_volatility_panel(
    *,
    ax,
    sigma: pd.Series,
    realized_vol: pd.Series,
    focus_asset: str,
    window: int,
    mdates,
    FormatStrFormatter,
) -> None:
    ax.plot(sigma.index, sigma, color="#ef4444", linewidth=1.5, label="Model σ")
    ax.plot(
        realized_vol.index,
        realized_vol,
        color="#67e8f9",
        linewidth=1.1,
        linestyle="--",
        alpha=0.9,
        label=f"Realized vol ({window}-day rolling)",
    )
    ax.fill_between(sigma.index, 0.0, sigma, color="#164e63", alpha=0.28)

    ax.set_title(
        f"{focus_asset}  —  conditional volatility",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        pad=8,
    )
    _style_axis(ax, formatter=FormatStrFormatter("%.3f"))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=9)


def _render_reliability_panel(
    *,
    ax,
    coverage: pd.DataFrame,
    FormatStrFormatter,
) -> None:
    nominal = coverage["nominal"].to_numpy()
    empirical = coverage["empirical"].to_numpy()
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#94a3b8",
        linestyle="--",
        linewidth=0.95,
        label="Calibrated (45°)",
    )
    ax.plot(
        nominal,
        empirical,
        color="#22d3ee",
        linewidth=1.4,
        marker="o",
        markersize=6,
        markerfacecolor="#22d3ee",
        markeredgecolor="#0e7490",
        label="Empirical coverage",
    )

    annotate_pairs = [(0.6827, "±1σ"), (0.9545, "±2σ")]
    for level, label in annotate_pairs:
        if not np.any(np.isclose(nominal, level)):
            continue
        idx = int(np.argmin(np.abs(nominal - level)))
        ax.annotate(
            f"{label}\nnom {nominal[idx]:.3f}\nemp {empirical[idx]:.3f}",
            xy=(nominal[idx], empirical[idx]),
            xytext=(10, -22),
            textcoords="offset points",
            color="#e5edf5",
            fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color="#475569", lw=0.7),
        )

    ax.set_title(
        "Reliability — nominal vs empirical coverage",
        loc="left",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        pad=8,
    )
    ax.set_xlabel("Nominal prediction-interval coverage", color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Empirical coverage", color="#cbd5e1", fontsize=10)
    ax.set_xlim(0.45, 1.01)
    ax.set_ylim(0.45, 1.01)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    _style_axis(ax, formatter=FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor="#e5edf5")


def _set_dark_style(plt) -> None:
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


def _style_axis(ax, *, formatter) -> None:
    ax.grid(axis="y", color="#223449", alpha=0.70, linewidth=0.7)
    ax.grid(axis="x", color="#162335", alpha=0.45, linewidth=0.6)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=10)
    ax.margins(x=0.01, y=0.06)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#42526b")
    ax.spines["bottom"].set_color("#42526b")


def _resolve_manifest_metadata(manifest_path: Path) -> tuple[str, str]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "ensemble_type" in raw:
        ensemble_manifest = load_panel_ensemble_manifest(manifest_path)
        if not ensemble_manifest.estimators:
            raise ValueError("Ensemble manifest contains no surviving estimators.")
        first_estimator = load_panel_manifest(
            Path(ensemble_manifest.estimators[0].estimator_manifest_path)
        )
        return first_estimator.target_column, (
            f"ensemble ({ensemble_manifest.aggregation}, n={len(ensemble_manifest.estimators)})"
        )
    manifest = load_panel_manifest(manifest_path)
    return manifest.target_column, manifest.model_name


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required_columns = {"asset_id", "forecast_start_date", "mu", "sigma"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Prediction CSV is missing required columns: {', '.join(missing_columns)}"
        )
    frame = frame.loc[:, ["asset_id", "forecast_start_date", "mu", "sigma"]].copy()
    frame["forecast_start_date"] = pd.to_datetime(frame["forecast_start_date"])
    return frame.set_index(["asset_id", "forecast_start_date"]).sort_index()


def _load_realized_series(
    *,
    csv_path: Path,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {date_column, id_column, target_column}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Realized dataset is missing required columns: {', '.join(missing_columns)}"
        )
    frame = frame.loc[:, [id_column, date_column, target_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.rename(columns={id_column: "asset_id", date_column: "forecast_start_date"})
    return frame.set_index(["asset_id", "forecast_start_date"]).sort_index()


def _compute_realized_volatility(*, y: pd.Series, window: int) -> pd.Series:
    return y.pow(2.0).rolling(window=window, min_periods=2).mean().pow(0.5)
