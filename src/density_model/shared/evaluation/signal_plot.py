"""
Cross-Sectional Signal Plot
---------------------------
Two-panel showcase of the model's *signal-extraction* output: the
predicted ``mu`` interpreted as a cross-sectional ranking signal.

For every forecast date the assets are z-scored on ``mu`` and bucketed
into ``n_buckets`` quantile groups. Within each bucket we average the
realized next-day return; pooled across all dates these means form the
per-quantile bar chart (left panel). The top-minus-bottom long/short
spread is then accumulated through time and plotted vs the realized
SPY benchmark (right panel) as the cleanest answer to *does the signal
rank returns?*.

Functions
---------
generate_signal_plot
    Entry point called by ``scripts/plot_signal.py``. Renders the PNG
    to ``output_path`` (defaults to
    ``<predictions-dir>/signal_decile.png``).
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

__all__ = ["generate_signal_plot"]

logger = logging.getLogger(__name__)


def generate_signal_plot(
    cfg: PanelPredictConfig,
    *,
    output_path: Path | None = None,
    signal_column: str = "mu",
    n_buckets: int = 5,
    benchmark_asset: str | None = "SPY",
    periods_per_year: int = 252,
) -> Path:
    """
    Render the two-panel cross-sectional signal showcase.

    Parameters
    ----------
    cfg : PanelPredictConfig
        Validated panel prediction config — same one used by
        ``predict.py`` / ``plot.py``.
    output_path : pathlib.Path | None
        Destination PNG. Defaults to ``<predictions-dir>/signal_decile.png``.
    signal_column : str
        Which column in the predictions CSV to z-score per date. Use
        ``"mu"`` for the directional signal, ``"mu_over_sigma"`` for
        the risk-adjusted variant (computed on the fly from ``mu`` and
        ``sigma``).
    n_buckets : int
        Number of cross-sectional quantile buckets. With 20 assets, 5
        is the natural choice (4 names per bucket per day).
    benchmark_asset : str | None
        Optional realized series to overlay on the cumulative spread
        panel (e.g. SPY). Pass ``None`` to suppress the benchmark line.
    periods_per_year : int
        Annualization factor for the bar-chart labels.

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
        raise ImportError("matplotlib must be installed to render the signal plot.") from error

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
        raise ValueError("Signal plot requires overlapping realized targets for forecast dates.")

    if output_path is None:
        output_path = cfg.output_path().parent / "signal_decile.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = joined.reset_index()
    if signal_column == "mu_over_sigma":
        df = df.loc[df["sigma"] > 0.0].copy()
        df["signal_raw"] = df["mu"] / df["sigma"]
        signal_label = "mu / sigma"
    elif signal_column == "mu":
        df = df.copy()
        df["signal_raw"] = df["mu"]
        signal_label = "mu"
    else:
        raise ValueError(f"signal_column must be 'mu' or 'mu_over_sigma'; got {signal_column!r}.")

    df = df.dropna(subset=["signal_raw", "target"])
    df["signal_z"] = df.groupby("forecast_start_date")["signal_raw"].transform(_zscore)
    df["bucket"] = df.groupby("forecast_start_date", group_keys=False).apply(
        lambda g: _assign_bucket(g["signal_z"], n_buckets=n_buckets), include_groups=False
    )
    df = df.dropna(subset=["bucket"])
    df["bucket"] = df["bucket"].astype(int)

    bucket_stats = _per_bucket_stats(df=df, periods_per_year=periods_per_year)
    spread, benchmark = _spread_and_benchmark(
        df=df,
        realized=realized.reset_index(),
        target_column=target_column,
        benchmark_asset=benchmark_asset,
        n_buckets=n_buckets,
    )

    _set_dark_style(plt)
    fig, (ax_bar, ax_spread) = plt.subplots(
        ncols=2,
        figsize=(15.5, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.55]},
    )
    fig.subplots_adjust(top=0.82, wspace=0.28)

    fig.suptitle(
        f"Signal extraction — cross-sectional ranking by {signal_label}",
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
        f"model={model_label}   |   {n_buckets} cross-sectional buckets per day "
        f"   |   {len(df):,} (asset, date) observations",
        ha="left",
        va="top",
        fontsize=10,
        color="#7dd3fc",
    )

    _render_bucket_panel(
        ax=ax_bar,
        bucket_stats=bucket_stats,
        n_buckets=n_buckets,
        FormatStrFormatter=FormatStrFormatter,
    )
    _render_spread_panel(
        ax=ax_spread,
        spread=spread,
        benchmark=benchmark,
        n_buckets=n_buckets,
        benchmark_asset=benchmark_asset,
        periods_per_year=periods_per_year,
        mdates=mdates,
        FormatStrFormatter=FormatStrFormatter,
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote signal plot to %s", output_path)
    return output_path


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0.0 or not np.isfinite(std):
        return series * 0.0
    return (series - series.mean()) / std


def _assign_bucket(signal_z: pd.Series, *, n_buckets: int) -> pd.Series:
    if signal_z.nunique() < n_buckets:
        return pd.Series(np.nan, index=signal_z.index)
    return pd.qcut(signal_z, q=n_buckets, labels=False, duplicates="drop") + 1


def _per_bucket_stats(*, df: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    grouped = df.groupby("bucket")["target"]
    stats = pd.DataFrame(
        {
            "mean_per_period": grouped.mean(),
            "n": grouped.size(),
        }
    )
    stats["annualized_return"] = stats["mean_per_period"] * float(periods_per_year)
    return stats.sort_index()


def _spread_and_benchmark(
    *,
    df: pd.DataFrame,
    realized: pd.DataFrame,
    target_column: str,
    benchmark_asset: str | None,
    n_buckets: int,
) -> tuple[pd.Series, pd.Series | None]:
    daily = df.groupby(["forecast_start_date", "bucket"])["target"].mean().unstack()
    if 1 not in daily.columns or n_buckets not in daily.columns:
        raise ValueError(
            "Spread plot requires at least one observation in both extreme buckets."
        )
    spread = (daily[n_buckets] - daily[1]).dropna()
    spread.name = "spread"

    benchmark = None
    if benchmark_asset is not None:
        bench_frame = realized.loc[realized["asset_id"] == benchmark_asset]
        if not bench_frame.empty:
            benchmark = (
                bench_frame.set_index("forecast_start_date")[target_column]
                .reindex(spread.index)
                .dropna()
            )
            benchmark.name = benchmark_asset

    return spread, benchmark


def _render_bucket_panel(
    *, ax, bucket_stats: pd.DataFrame, n_buckets: int, FormatStrFormatter
) -> None:
    buckets = bucket_stats.index.to_list()
    annualized_pct = bucket_stats["annualized_return"].to_numpy() * 100.0
    colors = _bucket_palette(len(buckets))
    bars = ax.bar(buckets, annualized_pct, color=colors, edgecolor="none", width=0.7)

    for bucket, value, bar in zip(buckets, annualized_pct, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + (0.4 if value >= 0 else -0.4),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            color="#e5edf5",
            fontsize=9.5,
        )

    bottom = float(annualized_pct[0])
    top = float(annualized_pct[-1])
    ax.set_title(
        f"Mean realized return per bucket  (Q{len(buckets)}−Q1 = {top - bottom:+.1f}%/yr)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
        pad=8,
    )
    ax.set_xlabel("Signal quantile bucket  (Q1 = lowest, Q{n} = highest)".format(n=n_buckets), color="#cbd5e1", fontsize=10)
    ax.set_ylabel("Annualized realized return", color="#cbd5e1", fontsize=10)
    ax.set_xticks(buckets)
    ax.set_xticklabels([f"Q{b}" for b in buckets])
    ax.axhline(0.0, color="#94a3b8", linewidth=0.85, alpha=0.9)
    _style_axis(ax, formatter=FormatStrFormatter("%.0f%%"))


def _render_spread_panel(
    *,
    ax,
    spread: pd.Series,
    benchmark: pd.Series | None,
    n_buckets: int,
    benchmark_asset: str | None,
    periods_per_year: int,
    mdates,
    FormatStrFormatter,
) -> None:
    cumulative_spread = (1.0 + spread).cumprod()
    cumulative_spread = pd.concat(
        [
            pd.Series([1.0], index=[cumulative_spread.index.min() - pd.Timedelta(days=1)]),
            cumulative_spread,
        ]
    )
    ax.plot(
        cumulative_spread.index,
        cumulative_spread,
        color="#22d3ee",
        linewidth=1.5,
        label=f"Q{n_buckets} − Q1 long/short (equal weight)",
    )

    if benchmark is not None and not benchmark.empty:
        cumulative_bench = (1.0 + benchmark).cumprod()
        cumulative_bench = pd.concat(
            [
                pd.Series([1.0], index=[cumulative_bench.index.min() - pd.Timedelta(days=1)]),
                cumulative_bench,
            ]
        )
        ax.plot(
            cumulative_bench.index,
            cumulative_bench,
            color="#f59e0b",
            linewidth=1.15,
            alpha=0.9,
            label=f"{benchmark_asset} (long-only reference)",
        )

    ax.axhline(1.0, color="#475569", linewidth=0.8, linestyle="--", alpha=0.9)

    sharpe = spread.mean() / spread.std() * np.sqrt(float(periods_per_year)) if spread.std() > 0 else 0.0
    annualized = spread.mean() * float(periods_per_year)
    ax.text(
        0.985,
        0.04,
        f"Spread Sharpe = {sharpe:+.2f}    |    Spread return = {annualized * 100.0:+.1f}% / yr",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#e5edf5",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#0b1728",
            edgecolor="#334155",
            linewidth=0.7,
        ),
    )

    ax.set_title(
        f"Cumulative growth — Q{n_buckets}−Q1 long/short",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
        pad=8,
    )
    ax.set_ylabel("Cumulative growth (×)", color="#cbd5e1", fontsize=10)
    _style_axis(ax, formatter=FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#e5edf5")

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis="x", labelsize=9)


def _bucket_palette(n: int) -> list[str]:
    base = np.array([14, 165, 233])  # #0ea5e9 blue
    top = np.array([34, 211, 238])  # #22d3ee cyan
    if n <= 1:
        return ["#22d3ee"]
    return [
        "#{:02x}{:02x}{:02x}".format(*[int(c) for c in (base + (top - base) * (i / (n - 1)))])
        for i in range(n)
    ]


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
