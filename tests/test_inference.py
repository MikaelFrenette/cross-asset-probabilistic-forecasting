"""
Inference Tests
---------------
Unit tests for manifest-driven prediction-time panel validation and XNYS
calendar alignment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import density_model.inference.forecasting as inference_module
from density_model.config import CsvDatasetConfig, TrainingArtifactManifest, load_train_config

__all__ = []


class FakeXNYSCalendar:
    """
    Test double for the XNYS calendar helper used by inference utilities.

    Parameters
    ----------
    calendar_name : str, default="XNYS"
        Requested calendar identifier.
    """

    def __init__(self, calendar_name: str = "XNYS") -> None:
        self.calendar_name = calendar_name

    def sessions_in_range(self, start_date, end_date) -> pd.DatetimeIndex:
        """
        Return business-day sessions for tests.

        Parameters
        ----------
        start_date : Any
            Inclusive start date.
        end_date : Any
            Inclusive end date.

        Returns
        -------
        pandas.DatetimeIndex
            Business-day sessions used by tests.
        """

        return pd.date_range(start=start_date, end=end_date, freq="B")


def build_manifest() -> TrainingArtifactManifest:
    """
    Build a minimal frozen training manifest for inference tests.

    Returns
    -------
    TrainingArtifactManifest
        Frozen training manifest.
    """

    config = load_train_config(Path("configs/train/main.yaml"))
    return TrainingArtifactManifest(
        feature_config=config.features,
        model_config_runtime=config.model,
        calendar="XNYS",
        preprocessing_path="artifacts/demo/preprocessing.json",
        model_state_path="artifacts/demo/model.pt",
    )


def test_normalize_prediction_panel_to_calendar_preserves_missing_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Align a raw inference panel to the master calendar and preserve missing rows.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the XNYS calendar helper.

    Returns
    -------
    None
        This test asserts calendar-normalized panel construction.
    """

    monkeypatch.setattr(inference_module, "XNYSCalendar", FakeXNYSCalendar)
    manifest = build_manifest()
    dataset_config = CsvDatasetConfig(csv_path="data/predict_features.csv", date_column="date", id_column="asset_id")
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-06", "2021-01-04", "2021-01-05", "2021-01-06"]),
            "asset_id": ["SPY", "SPY", "QQQ", "QQQ", "QQQ"],
            "return": [0.1, 0.3, 0.4, 0.5, 0.6],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ", "QQQ"],
        }
    )

    normalized = inference_module._normalize_prediction_panel_to_calendar(
        panel=panel,
        manifest=manifest,
        dataset_config=dataset_config,
    )

    spy_rows = normalized.loc[normalized["asset_id"] == "SPY"].reset_index(drop=True)
    assert len(spy_rows) == 3
    assert pd.isna(spy_rows.loc[1, "return"])
    assert spy_rows["ticker"].tolist() == ["SPY", "SPY", "SPY"]


def test_load_prediction_panel_rejects_missing_required_columns(tmp_path: Path) -> None:
    """
    Reject an external prediction dataset with missing required columns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test asserts fail-fast schema validation.
    """

    manifest = build_manifest()
    dataset_path = tmp_path / "prediction.csv"
    predict_config = type(
        "PredictConfigStub",
        (),
        {"dataset": CsvDatasetConfig(csv_path=str(dataset_path), date_column="date", id_column="asset_id")},
    )()
    pd.DataFrame(
        {
            "date": ["2021-01-04"],
            "asset_id": ["SPY"],
            "ticker": ["SPY"],
        }
    ).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        inference_module._load_prediction_panel(path=dataset_path, manifest=manifest, config=predict_config)


def test_build_rolling_prediction_panel_data_returns_multiple_forecast_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Build rolling forecast windows across multiple aligned forecast dates.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the XNYS calendar helper.

    Returns
    -------
    None
        This test asserts dataset-wide rolling inference construction.
    """

    monkeypatch.setattr(inference_module, "XNYSCalendar", FakeXNYSCalendar)
    manifest = build_manifest()
    dataset_config = CsvDatasetConfig(csv_path="data/predict_features.csv", date_column="date", id_column="asset_id")
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2021-01-07",
                    "2021-01-04",
                    "2021-01-05",
                    "2021-01-06",
                    "2021-01-07",
                ]
            ),
            "asset_id": ["SPY", "SPY", "SPY", "SPY", "QQQ", "QQQ", "QQQ", "QQQ"],
            "return": [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
            "ticker": ["SPY", "SPY", "SPY", "SPY", "QQQ", "QQQ", "QQQ", "QQQ"],
        }
    )

    manifest = manifest.model_copy(
        update={
            "feature_config": manifest.feature_config.model_copy(update={"sequence_length": 2, "forecast_horizon": 1})
        }
    )

    from density_model.preprocessing import PreprocessingBundle

    preprocessing = PreprocessingBundle().fit(
        panel,
        continuous_columns=("return",),
        categorical_columns=("ticker",),
    )

    panel_data = inference_module.build_rolling_prediction_panel_data(
        panel=panel,
        manifest=manifest,
        preprocessing=preprocessing,
        dataset_config=dataset_config,
    )

    assert panel_data["y"] is not None
    assert panel_data["y"].shape == (2, 2, 1, 1)
    assert list(pd.to_datetime(panel_data["forecast_start_dates"])) == [
        pd.Timestamp("2021-01-06"),
        pd.Timestamp("2021-01-07"),
    ]
