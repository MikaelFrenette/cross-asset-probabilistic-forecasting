"""
Quant Plotting
--------------
Generate presentation-oriented quant-style forecast reports with a cleaner
three-panel layout and compact summary statistics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from density_model.config import PlotConfig, load_training_manifest
from density_model.evaluation.plotting import (
    _compute_ewma_volatility,
    _compute_realized_volatility,
    _load_predictions,
    _load_realized_series,
)

__all__ = ["plot_quant_report_from_config"]


def plot_quant_report_from_config(*, config: PlotConfig) -> Path:
    """
    Generate per-asset quant-style forecast report plots from a rolling prediction CSV.

    Parameters
    ----------
    config : PlotConfig
        Typed plotting configuration.

    Returns
    -------
    pathlib.Path
        Output directory containing per-asset quant report plot files.
    """

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
    except ImportError as error:  # pragma: no cover - depends on optional plotting stack.
        raise ImportError("matplotlib must be installed to generate quant report plots.") from error

    manifest = load_training_manifest(config.manifest_file())
    target_column = manifest.feature_config.target_column
    predictions = _load_predictions(config.predictions_file())
    realized = _load_realized_series(
        csv_path=config.dataset.csv_file(),
        date_column=config.dataset.date_column,
        id_column=config.dataset.id_column,
        target_column=target_column,
    )

    prediction_dates = predictions.index.get_level_values("forecast_start_date")
    if prediction_dates.nunique() < 2:
        raise ValueError(
            "Quant-style plotting requires a rolling/backtest prediction file with multiple forecast dates per asset."
        )

    joined = predictions.join(realized.rename(columns={target_column: "target"}), how="inner")
    if joined.empty:
        raise ValueError("Quant-style plotting requires overlapping realized targets for the forecast dates.")

    output_dir = config.output_path() / "quant_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    for asset_id in joined.index.get_level_values("asset_id").unique():
        asset_frame = joined.xs(asset_id, level="asset_id").sort_index()
        y = asset_frame["target"]
        mu = asset_frame["mu"]
        sigma = asset_frame["sigma"]
        raw_signal = mu / sigma.replace(0.0, np.nan)
        signal = raw_signal.clip(lower=-3.0, upper=3.0)
        color_signal = _normalize_signal_for_color(raw_signal)
        ewma_vol = _compute_ewma_volatility(y=y).reindex(asset_frame.index)
        realized_vol = _compute_realized_volatility(y=y).reindex(asset_frame.index)

        summary = _summary_statistics(
            target=y,
            mu=mu,
            sigma=sigma,
            ewma_vol=ewma_vol,
            signal=signal,
        )

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
        ax_mean, ax_vol, ax_resid = axes

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
            f"Forecast window: {asset_frame.index.min().date()} to {asset_frame.index.max().date()}   "
            f"|   model={manifest.model_config_runtime.name}",
            ha="left",
            va="top",
            fontsize=10,
            color="#7dd3fc",
        )
        fig.text(
            0.94,
            0.955,
            _format_summary_block(summary),
            ha="right",
            va="top",
            fontsize=9.5,
            color="#f8fafc",
            family="DejaVu Sans Mono",
            bbox={"facecolor": "#12243a", "edgecolor": "#2f4968", "boxstyle": "square,pad=0.5"},
        )

        ax_mean.plot(y.index, y, color="#22d3ee", linewidth=0.8, alpha=0.88, label="Realized return")
        ax_mean.plot(mu.index, mu, color="#ef4444", linewidth=1.45, label="Model mean")
        ax_mean.fill_between(mu.index, mu - sigma, mu + sigma, color="#ef4444", alpha=0.22, label="±1σ band")
        ax_mean.fill_between(mu.index, mu - 2.0 * sigma, mu + 2.0 * sigma, color="#fb7185", alpha=0.10, label="±2σ band")

        ax_vol.plot(sigma.index, sigma, color="#ef4444", linewidth=1.45, label="Model sigma")
        ax_vol.plot(ewma_vol.index, ewma_vol, color="#818cf8", linewidth=1.15, alpha=0.95, label="EWMA volatility")
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
        ax_resid.axhline(0.0, color="#dce7f3", linewidth=0.95, alpha=0.9)
        ax_resid.axhline(1.0, color="#334155", linewidth=0.75, linestyle="--")
        ax_resid.axhline(-1.0, color="#334155", linewidth=0.75, linestyle="--")
        ax_resid.bar(
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
        _style_axis(ax_resid, "Risk-Adjusted Signal", formatter=FormatStrFormatter("%.2f"))

        ax_mean.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")
        ax_vol.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")
        ax_resid.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")

        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax_resid.xaxis.set_major_locator(locator)
        ax_resid.xaxis.set_major_formatter(formatter)
        ax_resid.tick_params(axis="x", labelsize=10)

        figure_path = output_dir / f"{asset_id}_quant.png"
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return output_dir


def _summary_statistics(
    *,
    target: pd.Series,
    mu: pd.Series,
    sigma: pd.Series,
    ewma_vol: pd.Series,
    signal: pd.Series,
) -> dict[str, float]:
    """
    Compute compact summary statistics for the quant-style forecast report.

    Parameters
    ----------
    target : pandas.Series
        Realized target series.
    mu : pandas.Series
        Predicted conditional mean series.
    sigma : pandas.Series
        Predicted conditional sigma series.
    ewma_vol : pandas.Series
        EWMA volatility reference series.
    signal : pandas.Series
        Risk-adjusted directional signal series defined as clipped ``mu / sigma``.

    Returns
    -------
    dict of str to float
        Compact scalar summary metrics used in the figure header.
    """

    valid = pd.concat(
        [
            target.rename("target"),
            mu.rename("mu"),
            sigma.rename("sigma"),
            ewma_vol.rename("ewma"),
            signal.rename("signal"),
        ],
        axis=1,
    ).dropna()
    mean_nll = float((((valid["target"] - valid["mu"]) ** 2) / (valid["sigma"] ** 2) + np.log(valid["sigma"] ** 2)).mean())
    mae = float((valid["target"] - valid["mu"]).abs().mean())
    sigma_mean = float(valid["sigma"].mean())
    ewma_corr = float(valid["sigma"].corr(valid["ewma"])) if len(valid) > 1 else float("nan")
    signal_mean = float(valid["signal"].mean())
    signal_abs_mean = float(valid["signal"].abs().mean())
    return {
        "mean_nll": mean_nll,
        "mae": mae,
        "sigma_mean": sigma_mean,
        "ewma_corr": ewma_corr,
        "signal_mean": signal_mean,
        "signal_abs_mean": signal_abs_mean,
    }


def _format_summary_block(summary: dict[str, float]) -> str:
    """
    Format compact summary statistics for the report header.

    Parameters
    ----------
    summary : dict of str to float
        Compact scalar summary metrics.

    Returns
    -------
    str
        Multi-line monospace summary block.
    """

    return "\n".join(
        [
            f"mean_nll   {summary['mean_nll']:+.4f}",
            f"mae        {summary['mae']:+.4f}",
            f"sigma_avg  {summary['sigma_mean']:+.4f}",
            f"corr_ewma  {summary['ewma_corr']:+.4f}",
            f"signal_mu  {summary['signal_mean']:+.4f}",
            f"|signal|   {summary['signal_abs_mean']:+.4f}",
        ]
    )


def _style_axis(ax, title: str, *, formatter) -> None:
    """
    Apply a cleaner quant-style axis treatment to one subplot.

    Parameters
    ----------
    ax : Any
        Matplotlib axis.
    title : str
        Axis title text.
    formatter : Any
        Matplotlib formatter applied to the y-axis.

    Returns
    -------
    None
        This function mutates the axis in place.
    """

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
    """
    Map a risk-adjusted signal value to a portfolio-style bar color.

    Parameters
    ----------
    value : float
        Clipped risk-adjusted signal value.

    Returns
    -------
    str
        Hex color string for the bar.
    """

    clipped = float(np.clip(value, -1.0, 1.0))
    neutral = np.array([148.0, 163.0, 184.0])
    positive = np.array([134.0, 239.0, 172.0])
    negative = np.array([252.0, 165.0, 165.0])
    target = positive if clipped >= 0.0 else negative
    blend = abs(clipped)
    rgb = (1.0 - blend) * neutral + blend * target
    return "#{:02x}{:02x}{:02x}".format(*(int(channel) for channel in rgb))


def _normalize_signal_for_color(signal: pd.Series) -> pd.Series:
    """
    Normalize a raw signal series for adaptive color intensity.

    The mapping is visualization-oriented: it rescales each asset's signal by a
    robust within-series magnitude estimate and then applies ``tanh`` so muted
    dummy signals still produce readable color separation without exploding
    outliers.

    Parameters
    ----------
    signal : pandas.Series
        Raw ``mu / sigma`` signal series.

    Returns
    -------
    pandas.Series
        Color-normalized signal in ``[-1, 1]``.
    """

    valid = signal.abs().dropna()
    if valid.empty:
        return signal * 0.0
    scale = float(valid.quantile(0.75))
    scale = max(scale, 0.05)
    return np.tanh(signal / scale)
