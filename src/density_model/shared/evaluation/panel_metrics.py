"""
Panel Forecast Metrics
----------------------
Density-forecast evaluation for panel predictions, extended to compare
the primary model against one or more benchmarks side-by-side.

For each model (primary + benchmarks):
  - Joins its ``predictions.csv`` with the realized target from the raw
    data CSV on ``(asset_id, forecast_start_date)``
  - Computes overall + per-asset metrics

When one of the benchmarks is designated the **reference** (typically
the unconditional-mean benchmark), RMSE skill scores are also reported:

    rmse_skill = rmse_model / rmse_reference

at both the overall (single scalar per model) and per-asset (one ratio
per asset per model) levels. Lower is better; <1 means the model beats
the reference on RMSE.

Density metrics (NLL, CRPS, coverage) are skipped for predictions whose
``sigma`` is NaN — the convention used by the point-only AR(1) and
unconditional-mean benchmarks.

Output files (written to ``cfg.output_path().parent``):
  - ``metrics.json``           — overall + per-asset payload for every model
  - ``metrics_per_asset.csv``  — long-format (asset_id, model, metrics)
  - ``metrics.md``             — markdown version of the stdout report

Functions
---------
generate_panel_metrics
    Entry point called by ``scripts/metrics.py``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from density_model.shared.config.panel_schema import PanelPredictConfig
from density_model.shared.evaluation.panel_plot import _resolve_manifest_metadata

__all__ = ["generate_panel_metrics"]

logger = logging.getLogger(__name__)

_SQRT_PI = math.sqrt(math.pi)


def generate_panel_metrics(
    cfg: PanelPredictConfig,
    *,
    benchmark_paths: Sequence[Path] | None = None,
    reference_path: Path | None = None,
) -> Path:
    """
    Evaluate the primary model and any benchmark predictions side-by-side.

    Parameters
    ----------
    cfg : PanelPredictConfig
        Same predict config consumed by ``scripts/predict.py`` /
        ``scripts/plot.py``; points at the primary model's manifest,
        predictions CSV, and the realized data CSV.
    benchmark_paths : sequence of pathlib.Path, optional
        Extra ``predictions.csv`` paths from benchmarks
        (``scripts/benchmark.py`` outputs). Each path's parent's last
        directory name is used as the model label
        (e.g. ``outputs/benchmarks/ar1/predictions.csv`` → ``ar1``).
    reference_path : pathlib.Path, optional
        A benchmark predictions CSV whose RMSE becomes the skill-score
        denominator for every model. Must also appear in
        ``benchmark_paths`` — or be the same path.
    """

    target_column, primary_label = _resolve_manifest_metadata(cfg.manifest_file())

    realized = _load_realized_series(
        csv_path=cfg.data.csv_file(),
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        target_column=target_column,
    )

    # Collect (label, predictions_path) pairs — primary first, benchmarks after.
    entries: list[tuple[str, Path]] = [(primary_label, cfg.output_path())]
    for path in benchmark_paths or []:
        entries.append((_infer_model_label(path), Path(path)))

    per_model: dict[str, dict[str, Any]] = {}
    for label, path in entries:
        predictions = _load_predictions(path)
        joined = predictions.join(
            realized.rename(columns={target_column: "target"}), how="inner"
        )
        if joined.empty:
            logger.warning("No overlapping realized targets for %s (%s) — skipping", label, path)
            continue
        joined = _compute_row_terms(joined)
        per_model[label] = {
            "path": str(path),
            "overall": _aggregate(joined),
            "per_asset": _aggregate_per_asset(joined),
            "n_rows": int(len(joined)),
        }

    if not per_model:
        raise ValueError("No models produced comparable metrics — check prediction paths.")

    # Resolve the reference model for RMSE skill scores.
    reference_label: str | None = None
    if reference_path is not None:
        ref_label = _infer_model_label(Path(reference_path))
        if ref_label not in per_model:
            # Reference may not have been passed via --benchmarks; load it directly.
            predictions = _load_predictions(Path(reference_path))
            joined = predictions.join(
                realized.rename(columns={target_column: "target"}), how="inner"
            )
            if joined.empty:
                raise ValueError(
                    f"Reference predictions {reference_path!r} share no overlap with realized data."
                )
            joined = _compute_row_terms(joined)
            per_model[ref_label] = {
                "path": str(reference_path),
                "overall": _aggregate(joined),
                "per_asset": _aggregate_per_asset(joined),
                "n_rows": int(len(joined)),
            }
        reference_label = ref_label

    if reference_label is not None:
        _attach_skill_scores(per_model, reference_label=reference_label)

    start_date, end_date, n_assets = _describe_range(per_model, primary_label)

    output_dir = cfg.output_path().parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json_path = output_dir / "metrics.json"
    metrics_csv_path = output_dir / "metrics_per_asset.csv"
    metrics_md_path = output_dir / "metrics.md"

    payload: dict[str, Any] = {
        "primary_model": primary_label,
        "target_column": target_column,
        "reference_model": reference_label,
        "date_range": [start_date.isoformat(), end_date.isoformat()],
        "n_assets": int(n_assets),
        "models": {
            label: {
                "path": info["path"],
                "n_rows": info["n_rows"],
                "overall": info["overall"],
                "per_asset": info["per_asset"].to_dict(orient="records"),
            }
            for label, info in per_model.items()
        },
    }
    metrics_json_path.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )

    long_frames: list[pd.DataFrame] = []
    for label, info in per_model.items():
        frame = info["per_asset"].copy()
        frame.insert(0, "model", label)
        long_frames.append(frame)
    long_frame = pd.concat(long_frames, ignore_index=True)
    long_frame.to_csv(metrics_csv_path, index=False)

    report = _render_report(
        primary_label=primary_label,
        reference_label=reference_label,
        per_model=per_model,
        start_date=start_date,
        end_date=end_date,
        n_assets=int(n_assets),
        output_dir=output_dir,
    )
    metrics_md_path.write_text(report + "\n", encoding="utf-8")
    print(report)

    logger.info(
        "Wrote metrics.json + metrics_per_asset.csv + metrics.md to %s", output_dir
    )
    return output_dir


def _compute_row_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row NLL / CRPS / error / coverage-indicator columns.

    Density-metric terms are NaN where ``sigma`` is NaN (point-only
    benchmarks); the aggregators use ``np.nanmean`` so those rows simply
    drop out of the density averages without affecting MAE/RMSE/bias.
    """

    frame = frame.copy()
    mu = frame["mu"].to_numpy(dtype=float)
    sigma = frame["sigma"].to_numpy(dtype=float)
    target = frame["target"].to_numpy(dtype=float)
    resid = target - mu
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_safe = np.where((~np.isnan(sigma)) & (sigma > 0.0), sigma, np.nan)
        frame["nll_row"] = (resid ** 2) / (sigma_safe ** 2) + np.log(sigma_safe ** 2)
        z = resid / sigma_safe
        frame["crps_row"] = sigma_safe * (
            z * (2.0 * _std_normal_cdf(z) - 1.0)
            + 2.0 * _std_normal_pdf(z)
            - 1.0 / _SQRT_PI
        )
        frame["cov_1s"] = (np.abs(resid) <= sigma_safe).astype(float)
        frame["cov_2s"] = (np.abs(resid) <= 2.0 * sigma_safe).astype(float)
        # Where sigma is NaN the comparison above resolves to False (0.0);
        # replace with NaN so nanmean correctly skips these rows.
        mask_nan = np.isnan(sigma_safe)
        frame.loc[mask_nan, "cov_1s"] = np.nan
        frame.loc[mask_nan, "cov_2s"] = np.nan

    frame["abs_err"] = np.abs(resid)
    frame["sq_err"] = resid ** 2
    frame["bias_row"] = mu - target
    return frame


def _aggregate(frame: pd.DataFrame) -> dict[str, float]:
    with np.errstate(invalid="ignore"):
        sigma_mean = (
            float(np.nanmean(frame["sigma"])) if frame["sigma"].notna().any() else float("nan")
        )
        return {
            "mean_nll": _safe_mean(frame["nll_row"]),
            "crps": _safe_mean(frame["crps_row"]),
            "mae_mu": _safe_mean(frame["abs_err"]),
            "rmse_mu": _safe_sqrt_mean(frame["sq_err"]),
            "bias_mu": _safe_mean(frame["bias_row"]),
            "coverage_1s": _safe_mean(frame["cov_1s"]),
            "coverage_2s": _safe_mean(frame["cov_2s"]),
            "sigma_mean": sigma_mean,
        }


def _aggregate_per_asset(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for asset_id, group in frame.groupby(level="asset_id"):
        stats = _aggregate(group)
        stats["asset_id"] = asset_id
        stats["n_rows"] = int(len(group))
        records.append(stats)
    per_asset = pd.DataFrame(records)
    column_order = [
        "asset_id",
        "n_rows",
        "mean_nll",
        "crps",
        "mae_mu",
        "rmse_mu",
        "bias_mu",
        "coverage_1s",
        "coverage_2s",
        "sigma_mean",
    ]
    per_asset = per_asset.loc[:, column_order]
    return per_asset.sort_values("asset_id").reset_index(drop=True)


def _attach_skill_scores(
    per_model: dict[str, dict[str, Any]], *, reference_label: str
) -> None:
    """Attach ``rmse_skill`` to every model's overall + per-asset metrics."""

    ref = per_model[reference_label]
    ref_overall_rmse = ref["overall"]["rmse_mu"]
    ref_per_asset = ref["per_asset"].set_index("asset_id")["rmse_mu"]

    for label, info in per_model.items():
        overall_rmse = info["overall"]["rmse_mu"]
        info["overall"]["rmse_skill"] = (
            float("nan")
            if ref_overall_rmse == 0 or not math.isfinite(ref_overall_rmse)
            else overall_rmse / ref_overall_rmse
        )
        per_asset = info["per_asset"].copy()
        denom = per_asset["asset_id"].map(ref_per_asset)
        per_asset["rmse_skill"] = per_asset["rmse_mu"] / denom.replace(0, np.nan)
        info["per_asset"] = per_asset


def _describe_range(
    per_model: dict[str, dict[str, Any]], primary_label: str
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Compute date bounds + asset count from the primary model's per-asset frame."""

    per_asset = per_model[primary_label]["per_asset"]
    n_assets = per_asset.shape[0]
    # Date bounds come from the primary model's predictions (joined).
    # Reconstruct by re-reading overall metrics — instead just expose
    # min/max via a fallback on any model's per-asset frame.
    # We stored row-level terms only per-model, so reconstruct from
    # each model's first/last predicted date via per_asset.n_rows is not
    # enough; instead we track overall range globally below.
    return _range_placeholder(per_model), _range_placeholder(per_model, mode="max"), n_assets


def _range_placeholder(
    per_model: dict[str, dict[str, Any]], *, mode: str = "min"
) -> pd.Timestamp:
    # Pull date bounds from the primary predictions file the first time
    # (lazy and cheap). If unavailable (no files readable), fall back to epoch.
    for info in per_model.values():
        try:
            frame = pd.read_csv(info["path"], usecols=["forecast_start_date"])
            series = pd.to_datetime(frame["forecast_start_date"])
            return pd.Timestamp(series.min() if mode == "min" else series.max())
        except Exception:
            continue
    return pd.Timestamp("1970-01-01")


def _render_report(
    *,
    primary_label: str,
    reference_label: str | None,
    per_model: dict[str, dict[str, Any]],
    start_date: Any,
    end_date: Any,
    n_assets: int,
    output_dir: Path,
) -> str:
    separator = "─" * 88
    lines: list[str] = []
    lines.append(f"Density Forecast Metrics — primary model: {primary_label}")
    lines.append(separator)
    lines.append(
        f"Date range : {pd.Timestamp(start_date).date()} – {pd.Timestamp(end_date).date()}   "
        f"|  {n_assets} assets"
    )
    if reference_label is not None:
        lines.append(f"Reference  : {reference_label}   (rmse_skill = rmse_model / rmse_reference)")
    else:
        lines.append("Reference  : (none — pass --reference to compute rmse_skill)")
    lines.append(separator)
    lines.append("Overall comparison (sorted with primary first, then benchmarks):")

    header = (
        f"  {'model':<28}{'n':>9}  {'NLL':>10}  {'CRPS':>10}  "
        f"{'MAE':>10}  {'RMSE':>10}  {'skill':>8}  {'bias':>11}  "
        f"{'cov_1σ':>8}  {'cov_2σ':>8}  {'σ̄':>10}"
    )
    lines.append(header)
    ordered_labels = [primary_label] + [lbl for lbl in per_model if lbl != primary_label]
    for label in ordered_labels:
        if label not in per_model:
            continue
        overall = per_model[label]["overall"]
        n_rows = per_model[label]["n_rows"]
        lines.append(_format_overall_row(label, overall, n_rows))
    lines.append(separator)

    if reference_label is not None:
        lines.append("Per-asset skill score (rmse_skill vs " f"{reference_label}; < 1 beats reference):")
        lines.append(_render_skill_table(per_model, primary_label=primary_label))
        lines.append(separator)

    lines.append(
        f"Wrote metrics.json + metrics_per_asset.csv + metrics.md to {output_dir}"
    )
    return "\n".join(lines)


def _format_overall_row(label: str, overall: dict[str, Any], n_rows: int) -> str:
    def fmt(value: float, spec: str = ".4f") -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "—"
        return f"{value:{spec}}"

    return (
        f"  {label:<28}{n_rows:>9}  "
        f"{fmt(overall['mean_nll'], '+.4f'):>10}  "
        f"{fmt(overall['crps'], '.6f'):>10}  "
        f"{fmt(overall['mae_mu'], '.6f'):>10}  "
        f"{fmt(overall['rmse_mu'], '.6f'):>10}  "
        f"{fmt(overall.get('rmse_skill'), '.3f'):>8}  "
        f"{fmt(overall['bias_mu'], '+.6f'):>11}  "
        f"{fmt(overall['coverage_1s'], '.4f'):>8}  "
        f"{fmt(overall['coverage_2s'], '.4f'):>8}  "
        f"{fmt(overall['sigma_mean'], '.6f'):>10}"
    )


def _render_skill_table(
    per_model: dict[str, dict[str, Any]], *, primary_label: str
) -> str:
    """Emit a compact per-asset table of rmse_skill for every model."""

    ordered_labels = [primary_label] + [lbl for lbl in per_model if lbl != primary_label]
    # Take the primary's asset order.
    primary_assets = per_model[primary_label]["per_asset"]["asset_id"].tolist()

    lines: list[str] = []
    header_parts = ["  ", f"{'asset_id':<10}"]
    for label in ordered_labels:
        header_parts.append(f"{label[:12]:>13}")
    lines.append("".join(header_parts))

    for asset_id in primary_assets:
        row_parts = ["  ", f"{str(asset_id):<10}"]
        for label in ordered_labels:
            per_asset = per_model[label]["per_asset"]
            match = per_asset.loc[per_asset["asset_id"] == asset_id]
            if match.empty or pd.isna(match["rmse_skill"].iloc[0]):
                row_parts.append(f"{'—':>13}")
            else:
                row_parts.append(f"{match['rmse_skill'].iloc[0]:>13.3f}")
        lines.append("".join(row_parts))
    return "\n".join(lines)


def _infer_model_label(path: Path) -> str:
    """Use ``parent_dir_name/predictions.csv`` convention to name a benchmark."""

    return path.parent.name or path.stem


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"asset_id", "forecast_start_date", "mu", "sigma"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Prediction CSV {path!s} is missing required columns: {', '.join(missing)}"
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
    required = {date_column, id_column, target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Realized dataset is missing required columns: {', '.join(missing)}"
        )
    frame = frame.loc[:, [id_column, date_column, target_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.rename(columns={id_column: "asset_id", date_column: "forecast_start_date"})
    return frame.set_index(["asset_id", "forecast_start_date"]).sort_index()


def _std_normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _std_normal_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf

    try:
        from scipy.special import erf as _erf  # type: ignore
    except ImportError:
        _erf = np.vectorize(erf)
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def _safe_mean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def _safe_sqrt_mean(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).any():
        return float("nan")
    return float(np.sqrt(np.nanmean(arr)))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON-serializable")
