"""
Temporal Integrity Tests
------------------------
Unit tests for anti-leakage temporal alignment in panel sequence generation.
"""

from __future__ import annotations

import pandas as pd

from density_model.datasets import VectorizedPanelDataset

__all__ = []


def test_vectorized_panel_target_starts_after_input_window() -> None:
    """
    Build a deterministic sequence and confirm targets begin strictly after inputs.

    Returns
    -------
    None
        This test asserts the panel vectorizer prevents direct look-ahead.
    """

    panel = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="B"),
            "asset_id": ["AAA"] * 6,
            "return": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "ticker": ["AAA"] * 6,
        }
    )

    panel_data = VectorizedPanelDataset(
        data=panel,
        date_column="date",
        id_column="asset_id",
        targets="return",
        in_steps=3,
        horizon=1,
        out_steps=1,
        dynamic_features=("return",),
        static_categorical_features=("ticker",),
    ).generate_sequences()

    assert panel_data["forecast_start_dates"][0] == pd.Timestamp("2024-01-04")
    assert panel_data["X_continuous"][0, 0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert panel_data["y"][0, 0, :, 0].tolist() == [3.0]
