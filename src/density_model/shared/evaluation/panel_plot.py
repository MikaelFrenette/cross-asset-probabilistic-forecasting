"""
Panel Forecast Plots
--------------------
Per-asset three-panel forecast reports for panel density-model outputs:

1. **Conditional Mean** — realized return vs model ``mu`` with ``±1σ`` and
   ``±2σ`` bands.
2. **Conditional Volatility** — model ``sigma`` overlaid with EWMA and
   rolling realized volatility references.
3. **Risk-Adjusted Signal** — clipped ``mu / sigma`` bars colored by
   direction and intensity.

The plot stack is purely a viewer: it reads ``predictions.csv`` (produced by
``scripts/predict.py``) and the raw realized-target CSV (the same
``data.csv_path`` used during inference), joins them by ``(asset_id,
forecast_start_date)``, and renders one PNG per asset in ``plots_dir``.

Supports single-model ``PanelTrainingManifest`` as well as ensemble
``PanelEnsembleManifest`` — the manifest is only read for ``target_column``.

Functions
---------
generate_panel_plots
    Entry point called by ``scripts/plot.py`` that runs the full render loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from density_model.shared.config.panel_schema import PanelPredictConfig
from density_model.shared.ensembling.panel.manifest import load_panel_ensemble_manifest
from density_model.shared.training.panel_manifest import load_panel_manifest

__all__ = ["generate_panel_plots"]

logger = logging.getLogger(__name__)


def generate_panel_plots(cfg: PanelPredictConfig) -> Path:
    """
    Render per-asset forecast report plots from a panel prediction config.

    Parameters
    ----------
    cfg : PanelPredictConfig
        Validated panel prediction configuration pointing at the manifest,
        realized data CSV, predictions CSV, and plot output directory.

    Returns
    -------
    pathlib.Path
        Output directory containing per-asset ``<asset_id>_report.png`` files.
    """

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except ImportError as error:
        raise ImportError("matplotlib must be installed to generate forecast plots.") from error

    target_column, model_label = _resolve_manifest_metadata(cfg.manifest_file())
    predictions = _load_predictions(cfg.output_path())
    realized = _load_realized_series(
        csv_path=cfg.data.csv_file(),
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        target_column=target_column,
    )

    prediction_dates = predictions.index.get_level_values("forecast_start_date")
    if prediction_dates.nunique() < 2:
        raise ValueError(
            "Plotting requires a rolling/backtest prediction file with multiple "
            "forecast dates per asset."
        )

    joined = predictions.join(realized.rename(columns={target_column: "target"}), how="inner")
    if joined.empty:
        raise ValueError("Plotting requires overlapping realized targets for the forecast dates.")

    output_dir = cfg.plots_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    for asset_id in joined.index.get_level_values("asset_id").unique():
        asset_frame = joined.xs(asset_id, level="asset_id").sort_index()
        y = asset_frame["target"]
        mu = asset_frame["mu"]
        sigma = asset_frame["sigma"]
        ewma_vol = _compute_ewma_volatility(y=y).reindex(asset_frame.index)
        realized_vol = _compute_realized_volatility(y=y).reindex(asset_frame.index)
        raw_signal = mu / sigma.replace(0.0, np.nan)
        signal = raw_signal.clip(lower=-3.0, upper=3.0)
        color_signal = _normalize_signal_for_color(raw_signal)

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

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(14, 10),
            gridspec_kw={"height_ratios": [2.2, 1.8, 1.4]},
            sharex=True,
        )
        fig.subplots_adjust(top=0.86, hspace=0.18)
        ax_mean, ax_vol, ax_signal = axes

        fig.suptitle(
            f"{asset_id}  |  Density Forecast Report",
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
            0.935,
            f"Forecast window: {asset_frame.index.min().date()} to "
            f"{asset_frame.index.max().date()}   |   model={model_label}",
            ha="left",
            va="top",
            fontsize=10,
            color="#7dd3fc",
        )
        ax_mean.plot(y.index, y, color="#22d3ee", linewidth=0.8, alpha=0.88, label="Realized return")
        ax_mean.plot(mu.index, mu, color="#ef4444", linewidth=1.45, label="Model mean")
        ax_mean.fill_between(
            mu.index, mu - sigma, mu + sigma, color="#ef4444", alpha=0.22, label="±1σ band"
        )
        ax_mean.fill_between(
            mu.index, mu - 2.0 * sigma, mu + 2.0 * sigma, color="#fb7185", alpha=0.10, label="±2σ band"
        )

        ax_vol.plot(sigma.index, sigma, color="#ef4444", linewidth=1.45, label="Model sigma")
        ax_vol.plot(
            ewma_vol.index, ewma_vol, color="#818cf8", linewidth=1.15, alpha=0.95, label="EWMA volatility"
        )
        ax_vol.plot(
            realized_vol.index,
            realized_vol,
            color="#67e8f9",
            linewidth=0.95,
            alpha=0.82,
            linestyle="--",
            label="Realized volatility",
        )
        ax_vol.fill_between(sigma.index, 0.0, sigma, color="#164e63", alpha=0.28)

        signal_colors = [_signal_color(value) for value in color_signal.fillna(0.0).tolist()]
        ax_signal.axhline(0.0, color="#dce7f3", linewidth=0.95, alpha=0.9)
        ax_signal.axhline(1.0, color="#334155", linewidth=0.75, linestyle="--")
        ax_signal.axhline(-1.0, color="#334155", linewidth=0.75, linestyle="--")
        ax_signal.bar(
            signal.index,
            signal.fillna(0.0),
            width=1.6,
            color=signal_colors,
            edgecolor="none",
            alpha=0.95,
            label="mu / sigma",
        )

        _style_axis(ax_mean, "Conditional Mean", formatter=FormatStrFormatter("%.3f"))
        _style_axis(ax_vol, "Conditional Volatility", formatter=FormatStrFormatter("%.3f"))
        _style_axis(ax_signal, "Risk-Adjusted Signal", formatter=FormatStrFormatter("%.2f"))

        ax_mean.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")
        ax_vol.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")
        ax_signal.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")

        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax_signal.xaxis.set_major_locator(locator)
        ax_signal.xaxis.set_major_formatter(formatter)
        ax_signal.tick_params(axis="x", labelsize=10)

        figure_path = output_dir / f"{asset_id}_report.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    logger.info("Wrote per-asset forecast reports to %s", output_dir)
    return output_dir


def _resolve_manifest_metadata(manifest_path: Path) -> tuple[str, str]:
    """Return ``(target_column, model_label)`` for both single-model and ensemble manifests."""

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


def _compute_realized_volatility(*, y: pd.Series, window: int = 20) -> pd.Series:
    return y.pow(2.0).rolling(window=window, min_periods=2).mean().pow(0.5)


def _compute_ewma_volatility(*, y: pd.Series, decay: float = 0.94) -> pd.Series:
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between 0 and 1.")
    return y.pow(2.0).ewm(alpha=1.0 - decay, adjust=False).mean().pow(0.5)


def _style_axis(ax, title: str, *, formatter) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#f8fafc", pad=8)
    ax.grid(axis="y", color="#223449", alpha=0.70, linewidth=0.7)
    ax.grid(axis="x", color="#162335", alpha=0.45, linewidth=0.6)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis="both", labelsize=10)
    ax.margins(x=0.01, y=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#42526b")
    ax.spines["bottom"].set_color("#42526b")


def _signal_color(value: float) -> str:
    clipped = float(np.clip(value, -1.0, 1.0))
    neutral = np.array([148.0, 163.0, 184.0])
    positive = np.array([134.0, 239.0, 172.0])
    negative = np.array([252.0, 165.0, 165.0])
    target = positive if clipped >= 0.0 else negative
    blend = abs(clipped)
    rgb = (1.0 - blend) * neutral + blend * target
    return "#{:02x}{:02x}{:02x}".format(*(int(channel) for channel in rgb))


def _normalize_signal_for_color(signal: pd.Series) -> pd.Series:
    valid = signal.abs().dropna()
    if valid.empty:
        return signal * 0.0
    scale = float(valid.quantile(0.75))
    scale = max(scale, 0.05)
    return np.tanh(signal / scale)
