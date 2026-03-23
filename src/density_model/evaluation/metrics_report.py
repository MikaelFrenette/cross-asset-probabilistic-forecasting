"""
Metrics Report
--------------
Generate conditional-mean evaluation tables using per-asset RMSE ratios
relative to a zero-return benchmark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "generate_metrics_report",
    "parse_model_spec",
    "_build_ar1_predictions",
    "_build_unconditional_mean_predictions",
    "_build_zero_benchmark_predictions",
    "_coverage",
    "_build_ratio_report",
    "_build_metric_sections",
]


DISPLAY_NAME_MAP = {
    "unconditional_mean": "Unconditional Mean",
    "ar1": "AR(1)",
    "cross_asset": "Cross-Asset Attention",
    "garch": "GARCH",
}


def generate_metrics_report(
    *,
    train_csv: str | Path,
    dataset_csv: str | Path,
    model_specs: list[str],
    target_column: str,
    date_column: str = "date",
    id_column: str = "asset_id",
    validation_csv: str | Path | None = None,
    output_dir: str | Path,
) -> Path:
    """
    Generate a conditional-mean RMSE ratio report against an AR(1) benchmark.
    Generate a conditional-mean RMSE ratio report against a zero-return
    benchmark.

    Parameters
    ----------
    train_csv : str or pathlib.Path
        Training feature CSV used to fit historical benchmark columns.
    dataset_csv : str or pathlib.Path
        Prediction-period feature CSV containing realized targets.
    model_specs : list of str
        Repeated ``name=prediction_csv`` specifications.
    target_column : str
        Target column used to evaluate conditional mean forecasts.
    date_column : str, default="date"
        Date column name.
    id_column : str, default="asset_id"
        Identifier column name.
    validation_csv : str or pathlib.Path or None, default=None
        Optional validation feature CSV appended to the benchmark history.
    output_dir : str or pathlib.Path
        Directory where report artifacts are written.

    Returns
    -------
    pathlib.Path
        Output directory containing the generated report files.
    """

    history = _load_target_panel(
        csv_path=train_csv,
        date_column=date_column,
        id_column=id_column,
        target_column=target_column,
    )
    if validation_csv is not None:
        validation = _load_target_panel(
            csv_path=validation_csv,
            date_column=date_column,
            id_column=id_column,
            target_column=target_column,
        )
        history = pd.concat([history, validation], ignore_index=True)

    realized = _load_target_panel(
        csv_path=dataset_csv,
        date_column=date_column,
        id_column=id_column,
        target_column=target_column,
    )
    baseline_predictions = _build_zero_benchmark_predictions(realized=realized, date_column=date_column, id_column=id_column)

    model_frames: dict[str, pd.DataFrame] = {}
    model_frames["unconditional_mean"] = _build_unconditional_mean_predictions(
        history=history,
        realized=realized,
        date_column=date_column,
        id_column=id_column,
        target_column=target_column,
    )
    model_frames["ar1"] = _build_ar1_predictions(
        history=history,
        realized=realized,
        date_column=date_column,
        id_column=id_column,
        target_column=target_column,
    )
    for model_spec in model_specs:
        model_name, prediction_path = parse_model_spec(model_spec)
        model_frames[model_name] = _load_prediction_frame(Path(prediction_path))

    report = _build_ratio_report(
        realized=realized,
        baseline_predictions=baseline_predictions,
        model_predictions=model_frames,
        date_column=date_column,
        id_column=id_column,
        target_column=target_column,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path / "rmse_ratio_report.csv", index=False)
    for metric_name, metric_frame in _build_metric_sections(report).items():
        metric_frame.to_csv(output_path / f"{metric_name}.csv", index=False)
    (output_path / "results.md").write_text(_format_markdown_report(report), encoding="utf-8")
    return output_path


def parse_model_spec(model_spec: str) -> tuple[str, Path]:
    """
    Parse one ``name=prediction_csv`` model specification.

    Parameters
    ----------
    model_spec : str
        Model specification string.

    Returns
    -------
    tuple of (str, pathlib.Path)
        Parsed model name and prediction CSV path.
    """

    if "=" not in model_spec:
        raise ValueError("Model specs must look like 'model_name=path/to/predictions.csv'.")
    model_name, raw_path = model_spec.split("=", 1)
    model_name = model_name.strip()
    raw_path = raw_path.strip()
    if not model_name or not raw_path:
        raise ValueError("Model specs must provide both a model name and a prediction CSV path.")
    return model_name, Path(raw_path)


def _load_target_panel(
    *,
    csv_path: str | Path,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Load one realized target panel from a feature CSV.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Feature CSV path.
    date_column : str
        Date column name.
    id_column : str
        Identifier column name.
    target_column : str
        Target column name.

    Returns
    -------
    pandas.DataFrame
        Clean realized target panel.
    """

    frame = pd.read_csv(csv_path)
    required_columns = {date_column, id_column, target_column}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing_text}")
    frame = frame.loc[:, [date_column, id_column, target_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    frame = frame.loc[frame[target_column].notna()].copy()
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)


def _load_prediction_frame(path: Path) -> pd.DataFrame:
    """
    Load one model prediction CSV.

    Parameters
    ----------
    path : pathlib.Path
        Prediction CSV path.

    Returns
    -------
    pandas.DataFrame
        Clean conditional mean prediction frame.
    """

    frame = pd.read_csv(path)
    required_columns = {"forecast_start_date", "asset_id", "mu"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Prediction CSV is missing required columns: {missing_text}")
    selected_columns = ["forecast_start_date", "asset_id", "mu"]
    if "sigma" in frame.columns:
        selected_columns.append("sigma")
    frame = frame.loc[:, selected_columns].copy()
    frame["forecast_start_date"] = pd.to_datetime(frame["forecast_start_date"])
    renamed = frame.rename(columns={"forecast_start_date": "date", "asset_id": "asset_id", "mu": "prediction"})
    if "sigma" not in renamed.columns:
        renamed["sigma"] = np.nan
    return renamed


def _build_zero_benchmark_predictions(
    *,
    realized: pd.DataFrame,
    date_column: str,
    id_column: str,
) -> pd.DataFrame:
    """
    Build zero-return benchmark predictions for the evaluation period.

    Parameters
    ----------
    realized : pandas.DataFrame
        Evaluation-period realized target panel.
    date_column : str
        Date column name.
    id_column : str
        Identifier column name.

    Returns
    -------
    pandas.DataFrame
        Long-form zero-return prediction frame with columns ``date``,
        ``asset_id``, and ``prediction``.
    """

    frame = realized.loc[:, [date_column, id_column]].copy()
    frame = frame.rename(columns={date_column: "date", id_column: "asset_id"})
    frame["asset_id"] = frame["asset_id"].astype(str)
    frame["prediction"] = 0.0
    frame["sigma"] = np.nan
    return frame.sort_values(["asset_id", "date"]).reset_index(drop=True)


def _build_unconditional_mean_predictions(
    *,
    history: pd.DataFrame,
    realized: pd.DataFrame,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Build per-stock unconditional-mean predictions for the evaluation period.

    Parameters
    ----------
    history : pandas.DataFrame
        Historical panel used to estimate per-stock unconditional means.
    realized : pandas.DataFrame
        Evaluation-period realized target panel.
    date_column : str
        Date column name.
    id_column : str
        Identifier column name.
    target_column : str
        Target column name.

    Returns
    -------
    pandas.DataFrame
        Long-form unconditional-mean prediction frame.
    """

    means = (
        history.groupby(id_column, sort=True)[target_column]
        .mean()
        .rename("prediction")
        .reset_index()
        .rename(columns={id_column: "asset_id"})
    )
    means["asset_id"] = means["asset_id"].astype(str)
    frame = realized.loc[:, [date_column, id_column]].copy()
    frame = frame.rename(columns={date_column: "date", id_column: "asset_id"})
    frame["asset_id"] = frame["asset_id"].astype(str)
    frame = frame.merge(means, on="asset_id", how="left")
    frame["sigma"] = np.nan
    return frame.sort_values(["asset_id", "date"]).reset_index(drop=True)


def _build_ar1_predictions(
    *,
    history: pd.DataFrame,
    realized: pd.DataFrame,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Build one-step-ahead AR(1) predictions for the evaluation period.

    Parameters
    ----------
    history : pandas.DataFrame
        Historical panel used to fit per-stock AR(1) coefficients.
    realized : pandas.DataFrame
        Evaluation-period realized target panel.
    date_column : str
        Date column name.
    id_column : str
        Identifier column name.
    target_column : str
        Target column name.

    Returns
    -------
    pandas.DataFrame
        Long-form AR(1) prediction frame.
    """

    rows: list[dict[str, object]] = []
    for asset_id, history_group in history.groupby(id_column, sort=True):
        history_series = history_group.sort_values(date_column).set_index(date_column)[target_column].dropna()
        realized_group = (
            realized.loc[realized[id_column].astype(str) == str(asset_id)]
            .sort_values(date_column)
            .set_index(date_column)[target_column]
            .dropna()
        )
        if history_series.empty or realized_group.empty:
            continue
        alpha, beta = _fit_ar1_coefficients(history_series)
        previous_value = float(history_series.iloc[-1])
        for forecast_date, observed_value in realized_group.items():
            rows.append(
                {
                    "date": pd.Timestamp(forecast_date),
                    "asset_id": str(asset_id),
                    "prediction": float(alpha + beta * previous_value),
                    "sigma": np.nan,
                }
            )
            previous_value = float(observed_value)
    return pd.DataFrame(rows)


def _fit_ar1_coefficients(series: pd.Series) -> tuple[float, float]:
    """
    Fit AR(1) coefficients by ordinary least squares.

    Parameters
    ----------
    series : pandas.Series
        Historical target series.

    Returns
    -------
    tuple of float
        Estimated intercept and lag coefficient.
    """

    lagged = series.shift(1)
    valid = pd.DataFrame({"y": series, "lagged": lagged}).dropna()
    if valid.empty:
        return 0.0, 0.0
    x = valid["lagged"].to_numpy(dtype=float)
    y = valid["y"].to_numpy(dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(((x - x_mean) ** 2).sum())
    if denominator <= 0.0:
        return y_mean, 0.0
    beta = float(((x - x_mean) * (y - y_mean)).sum() / denominator)
    alpha = float(y_mean - beta * x_mean)
    return alpha, beta


def _build_ratio_report(
    *,
    realized: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    model_predictions: dict[str, pd.DataFrame],
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Build the per-stock RMSE ratio report against the AR(1) baseline.
    Build the per-stock RMSE ratio report against the zero-return benchmark.

    Parameters
    ----------
    realized : pandas.DataFrame
        Evaluation-period realized target panel.
    baseline_predictions : pandas.DataFrame
        Unconditional-mean baseline predictions.
    model_predictions : dict of str to pandas.DataFrame
        Named model prediction frames.
    date_column : str
        Date column name in the realized panel.
    id_column : str
        Identifier column name in the realized panel.
    target_column : str
        Target column name in the realized panel.

    Returns
    -------
    pandas.DataFrame
        Report frame with one row per asset plus a weighted-average row.
    """

    base = realized.rename(columns={date_column: "date", id_column: "asset_id", target_column: "target"}).copy()
    base["asset_id"] = base["asset_id"].astype(str)
    base = base.merge(baseline_predictions, on=["date", "asset_id"], how="inner", suffixes=("", "_baseline"))
    base = base.rename(columns={"prediction": "prediction_baseline"})

    metric_names = ["rmse_ratio_to_zero", "coverage_68", "coverage_95"]
    report_rows: list[dict[str, object]] = []
    asset_ids = sorted(base["asset_id"].unique().tolist())
    weighted_numerators = {
        metric_name: {model_name: 0.0 for model_name in model_predictions} for metric_name in metric_names
    }
    weighted_counts = {
        metric_name: {model_name: 0 for model_name in model_predictions} for metric_name in metric_names
    }

    for asset_id in asset_ids:
        asset_panel = base.loc[base["asset_id"] == asset_id].copy()
        for model_name, prediction_frame in model_predictions.items():
            asset_prediction = prediction_frame.loc[prediction_frame["asset_id"].astype(str) == asset_id].copy()
            if "sigma" not in asset_prediction.columns:
                asset_prediction["sigma"] = np.nan
            asset_prediction = asset_prediction.rename(
                columns={"prediction": f"prediction_{model_name}", "sigma": f"sigma_{model_name}"}
            )
            asset_panel = asset_panel.merge(
                asset_prediction.loc[:, ["date", "asset_id", f"prediction_{model_name}", f"sigma_{model_name}"]],
                on=["date", "asset_id"],
                how="inner",
            )
        if asset_panel.empty:
            continue
        baseline_rmse = _rmse(asset_panel["target"], asset_panel["prediction_baseline"])
        if not np.isfinite(baseline_rmse) or baseline_rmse <= 0.0:
            continue
        metric_rows = {metric_name: {"metric": metric_name, "row": asset_id} for metric_name in metric_names}
        for model_name in model_predictions:
            model_rmse = _rmse(asset_panel["target"], asset_panel[f"prediction_{model_name}"])
            ratio = model_rmse / baseline_rmse if np.isfinite(model_rmse) else np.nan
            metric_rows["rmse_ratio_to_zero"][model_name] = ratio
            if np.isfinite(ratio):
                weighted_numerators["rmse_ratio_to_zero"][model_name] += float(ratio) * len(asset_panel)
                weighted_counts["rmse_ratio_to_zero"][model_name] += len(asset_panel)

            coverage_68 = _coverage(asset_panel["target"], asset_panel[f"prediction_{model_name}"], asset_panel[f"sigma_{model_name}"], z_value=1.0)
            metric_rows["coverage_68"][model_name] = coverage_68
            if np.isfinite(coverage_68):
                weighted_numerators["coverage_68"][model_name] += float(coverage_68) * len(asset_panel)
                weighted_counts["coverage_68"][model_name] += len(asset_panel)

            coverage_95 = _coverage(asset_panel["target"], asset_panel[f"prediction_{model_name}"], asset_panel[f"sigma_{model_name}"], z_value=1.96)
            metric_rows["coverage_95"][model_name] = coverage_95
            if np.isfinite(coverage_95):
                weighted_numerators["coverage_95"][model_name] += float(coverage_95) * len(asset_panel)
                weighted_counts["coverage_95"][model_name] += len(asset_panel)
        report_rows.extend(metric_rows.values())

    for metric_name in metric_names:
        average_row: dict[str, object] = {"metric": metric_name, "row": "avg"}
        for model_name in model_predictions:
            if weighted_counts[metric_name][model_name] == 0:
                average_row[model_name] = np.nan
            else:
                average_row[model_name] = weighted_numerators[metric_name][model_name] / weighted_counts[metric_name][model_name]
        report_rows.append(average_row)
    return pd.DataFrame(report_rows)


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute RMSE between two aligned series.

    Parameters
    ----------
    y_true : pandas.Series
        Realized target values.
    y_pred : pandas.Series
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error.
    """

    if len(y_true) == 0 or len(y_pred) == 0:
        return float("nan")
    residual = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def _coverage(y_true: pd.Series, mu: pd.Series, sigma: pd.Series, *, z_value: float) -> float:
    """
    Compute empirical interval coverage for a Gaussian-style predictive band.

    Parameters
    ----------
    y_true : pandas.Series
        Realized targets.
    mu : pandas.Series
        Predicted conditional means.
    sigma : pandas.Series
        Predicted conditional standard deviations.
    z_value : float
        Band half-width multiplier, e.g. ``1.0`` or ``1.96``.

    Returns
    -------
    float
        Empirical coverage rate, or ``nan`` when ``sigma`` is unavailable.
    """

    valid = sigma.notna() & (sigma > 0.0)
    if not valid.any():
        return float("nan")
    y = y_true.loc[valid].to_numpy(dtype=float)
    center = mu.loc[valid].to_numpy(dtype=float)
    scale = sigma.loc[valid].to_numpy(dtype=float)
    lower = center - z_value * scale
    upper = center + z_value * scale
    inside = (y >= lower) & (y <= upper)
    return float(np.mean(inside))


def _format_markdown_report(report: pd.DataFrame) -> str:
    """
    Format the metrics report as grouped markdown sections.

    Parameters
    ----------
    report : pandas.DataFrame
        Report frame from ``_build_ratio_report``.

    Returns
    -------
    str
        Markdown report text grouped by metric blocks.
    """

    value_columns = [column for column in report.columns if column not in {"metric", "row"}]
    display_columns = [DISPLAY_NAME_MAP.get(column, column.replace("_", " ").title()) for column in value_columns]
    metric_labels = {
        "rmse_ratio_to_zero": "RMSE",
        "coverage_68": "Coverage 68%",
        "coverage_95": "Coverage 95%",
    }
    lines = [
        "# Results",
        "",
        "All values are evaluated on the common per-asset overlap across the listed columns.",
        "RMSE is reported as a ratio to the zero-return benchmark.",
        "",
    ]
    for metric_name, group in report.groupby("metric", sort=False):
        title = metric_labels.get(metric_name, metric_name)
        lines.append(f"{title}:")
        lines.append("")
        row_label_width = max(len("asset"), max(len(str(value)) for value in group["row"]))
        value_width = 8
        header = "  " + "asset".ljust(row_label_width) + " | " + " | ".join(
            column.ljust(max(len(column), value_width)) for column in display_columns
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for _, row in group.iterrows():
            row_name = str(row["row"]).ljust(row_label_width)
            value_parts: list[str] = []
            for column in value_columns:
                value = row[column]
                formatted_value = "" if pd.isna(value) else f"{float(value):.3f}"
                value_parts.append(formatted_value.ljust(max(len(column), value_width)))
            lines.append(f"  {row_name} | " + " | ".join(value_parts).rstrip())
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_metric_sections(report: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build one wide CSV-ready frame per metric section.

    Parameters
    ----------
    report : pandas.DataFrame
        Long-form report frame from ``_build_ratio_report``.

    Returns
    -------
    dict of str to pandas.DataFrame
        Mapping from metric name to wide per-asset report frame.
    """

    section_frames: dict[str, pd.DataFrame] = {}
    value_columns = [column for column in report.columns if column not in {"metric", "row"}]
    for metric_name, group in report.groupby("metric", sort=False):
        section = group.loc[:, ["row", *value_columns]].copy()
        renamed_columns = {"row": "asset"}
        renamed_columns.update(
            {column: DISPLAY_NAME_MAP.get(column, column.replace("_", " ").title()) for column in value_columns}
        )
        section = section.rename(columns=renamed_columns)
        section_frames[metric_name] = section.reset_index(drop=True)
    return section_frames
