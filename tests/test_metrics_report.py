"""
Metrics Report Tests
--------------------
Unit tests for zero-benchmark conditional-mean evaluation reports.
"""

from __future__ import annotations

import pandas as pd

from density_model.evaluation.metrics_report import (
    _coverage,
    _build_ar1_predictions,
    _build_unconditional_mean_predictions,
    _build_zero_benchmark_predictions,
    _build_ratio_report,
    parse_model_spec,
)

__all__ = []


def test_parse_model_spec() -> None:
    """
    Parse one ``name=path`` model specification.

    Returns
    -------
    None
        This test asserts public CLI model spec parsing.
    """

    name, path = parse_model_spec("cross_asset=artifacts/demo/predictions.csv")
    assert name == "cross_asset"
    assert str(path).endswith("predictions.csv")


def test_build_ratio_report_returns_weighted_average_row() -> None:
    """
    Build a simple per-stock zero-benchmark-relative RMSE report.

    Returns
    -------
    None
        This test asserts report structure and weighted average generation.
    """

    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-01", "2020-01-02", "2020-01-03"]
            ),
            "asset_id": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "return": [0.0, 1.0, 2.0, 0.0, -1.0, -2.0],
        }
    )
    realized = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-06", "2020-01-07"]),
            "asset_id": ["AAA", "AAA", "BBB", "BBB"],
            "return": [2.7, 4.1, -2.8, -4.2],
        }
    )
    baseline = _build_zero_benchmark_predictions(
        realized=realized,
        date_column="date",
        id_column="asset_id",
    )
    unconditional_mean = _build_unconditional_mean_predictions(
        history=history,
        realized=realized,
        date_column="date",
        id_column="asset_id",
        target_column="return",
    )
    ar1 = _build_ar1_predictions(
        history=history,
        realized=realized,
        date_column="date",
        id_column="asset_id",
        target_column="return",
    )
    model_predictions = {
        "unconditional_mean": unconditional_mean,
        "ar1": ar1,
        "perfect": pd.DataFrame(
                {
                    "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-06", "2020-01-07"]),
                    "asset_id": ["AAA", "AAA", "BBB", "BBB"],
                    "prediction": [2.7, 4.1, -2.8, -4.2],
                }
            )
        }

    report = _build_ratio_report(
        realized=realized,
        baseline_predictions=baseline,
        model_predictions=model_predictions,
        date_column="date",
        id_column="asset_id",
        target_column="return",
    )

    assert report["metric"].tolist() == [
        "rmse_ratio_to_zero",
        "coverage_68",
        "coverage_95",
        "rmse_ratio_to_zero",
        "coverage_68",
        "coverage_95",
        "rmse_ratio_to_zero",
        "coverage_68",
        "coverage_95",
    ]
    assert "unconditional_mean" in report.columns
    assert "ar1" in report.columns
    aaa_rmse_row = report.loc[(report["metric"] == "rmse_ratio_to_zero") & (report["row"] == "AAA")].iloc[0]
    avg_rmse_row = report.loc[(report["metric"] == "rmse_ratio_to_zero") & (report["row"] == "avg")].iloc[0]
    aaa_cov68_row = report.loc[(report["metric"] == "coverage_68") & (report["row"] == "AAA")].iloc[0]
    assert float(aaa_rmse_row["perfect"]) == 0.0
    assert float(avg_rmse_row["perfect"]) == 0.0
    assert pd.isna(aaa_cov68_row["unconditional_mean"])


def test_ratio_report_uses_common_overlap_across_columns() -> None:
    """
    Score all report columns on the common per-asset overlap, not each model's own window.

    Returns
    -------
    None
        This test asserts cross-model comparisons are date-aligned.
    """

    realized = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
            "asset_id": ["AAA", "AAA", "AAA"],
            "return": [1.0, 2.0, 3.0],
        }
    )
    baseline = _build_zero_benchmark_predictions(
        realized=realized,
        date_column="date",
        id_column="asset_id",
    )
    model_predictions = {
        "early_model": pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
                "asset_id": ["AAA", "AAA", "AAA"],
                "prediction": [1.0, 2.0, 3.0],
            }
        ),
        "late_model": pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-08"]),
                "asset_id": ["AAA"],
                "prediction": [0.0],
            }
        ),
    }

    report = _build_ratio_report(
        realized=realized,
        baseline_predictions=baseline,
        model_predictions=model_predictions,
        date_column="date",
        id_column="asset_id",
        target_column="return",
    )

    aaa_row = report.loc[(report["metric"] == "rmse_ratio_to_zero") & (report["row"] == "AAA")].iloc[0]
    assert float(aaa_row["early_model"]) == 0.0
    assert float(aaa_row["late_model"]) == 1.0


def test_coverage_computes_empirical_interval_hit_rate() -> None:
    """
    Compute Gaussian-style interval coverage from mean and sigma forecasts.

    Returns
    -------
    None
        This test asserts empirical interval coverage is computed correctly.
    """

    y_true = pd.Series([0.0, 1.0, 2.0])
    mu = pd.Series([0.0, 0.0, 0.0])
    sigma = pd.Series([1.0, 1.0, 1.0])

    coverage_68 = _coverage(y_true, mu, sigma, z_value=1.0)
    coverage_95 = _coverage(y_true, mu, sigma, z_value=1.96)

    assert coverage_68 == 2 / 3
    assert coverage_95 == 2 / 3
