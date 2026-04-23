"""
Panel Preprocess Runner
-----------------------
Standalone preprocessing step for panel forecasting pipelines: load the train
and validation CSVs, fit the :class:`PreprocessingBundle` on the training
panel, transform both splits, vectorize them into ``(B, ID, T, K)`` tensors,
and persist everything to the configured artifacts directory.

Functions
---------
run_panel_preprocess
    Top-level entry point called by ``scripts/preprocess.py``. Writes:

    - ``preprocessing.json`` — fitted bundle (serialized)
    - ``train_panel.pt`` — vectorized training panel + metadata
    - ``val_panel.pt`` — vectorized validation panel + metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from dlecosys.shared.config.panel_schema import PanelPipelineConfig
from dlecosys.shared.data.panel.vectorizer import VectorizedPanelDataset
from dlecosys.shared.preprocessing.panel import PreprocessingBundle

__all__ = ["run_panel_preprocess", "train_panel_path", "val_panel_path"]

logger = logging.getLogger(__name__)


def train_panel_path(cfg: PanelPipelineConfig) -> Path:
    """Return the resolved path for the vectorized training panel artifact."""

    return cfg.artifacts.output_path() / "train_panel.pt"


def val_panel_path(cfg: PanelPipelineConfig) -> Path:
    """Return the resolved path for the vectorized validation panel artifact."""

    return cfg.artifacts.output_path() / "val_panel.pt"


def run_panel_preprocess(cfg: PanelPipelineConfig, *, overwrite: bool = False) -> Path:
    """
    Load raw CSVs, fit the preprocessing bundle, and write vectorized panels.

    Parameters
    ----------
    cfg : PanelPipelineConfig
        Validated top-level panel pipeline configuration.
    overwrite : bool, default=False
        When ``False``, raise if any target artifact already exists.

    Returns
    -------
    pathlib.Path
        The artifacts directory containing the written preprocessing bundle
        and vectorized panels.
    """

    output_dir = cfg.artifacts.output_path().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        cfg.artifacts.preprocessing_path().resolve(),
        train_panel_path(cfg).resolve(),
        val_panel_path(cfg).resolve(),
    ]
    if not overwrite:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing artifact(s): "
                + ", ".join(existing)
                + " (pass --overwrite to allow)."
            )

    logger.info("Loading train + val feature panels")
    required_columns = _required_columns(cfg)
    train_panel = _load_feature_panel(
        csv_path=cfg.data.train_csv_path(),
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        required_columns=required_columns,
    )
    val_panel = _load_feature_panel(
        csv_path=cfg.data.val_csv_path(),
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        required_columns=required_columns,
    )

    logger.info("Fitting preprocessing bundle on training panel")
    preprocessing = PreprocessingBundle(id_column=cfg.data.id_column).fit(
        train_panel,
        continuous_columns=cfg.features.continuous_columns,
        categorical_columns=(
            cfg.features.dynamic_categorical_columns
            + cfg.features.static_categorical_columns
        ),
    )

    logger.info("Vectorizing train + val panels")
    train_vectorized = _vectorize(preprocessing.transform(train_panel), cfg)
    val_vectorized = _vectorize(preprocessing.transform(val_panel), cfg)
    if train_vectorized["y"] is None:
        raise ValueError("Training dataset did not produce any valid forecast windows.")
    if val_vectorized["y"] is None:
        raise ValueError("Validation dataset did not produce any valid forecast windows.")

    preprocessing_path = cfg.artifacts.preprocessing_path().resolve()
    preprocessing.save(preprocessing_path)
    torch.save(train_vectorized, train_panel_path(cfg).resolve())
    torch.save(val_vectorized, val_panel_path(cfg).resolve())

    logger.info(
        "Wrote preprocessing bundle (%s) and vectorized panels to %s",
        preprocessing_path,
        output_dir,
    )
    return output_dir


def _required_columns(cfg: PanelPipelineConfig) -> set[str]:
    return {
        cfg.data.date_column,
        cfg.data.id_column,
        cfg.features.target_column,
        *cfg.features.continuous_columns,
        *cfg.features.dynamic_categorical_columns,
        *cfg.features.static_categorical_columns,
    }


def _load_feature_panel(
    *,
    csv_path: Path,
    date_column: str,
    id_column: str,
    required_columns: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Panel CSV {csv_path} is missing required columns: {', '.join(missing)}"
        )
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)


def _vectorize(panel: pd.DataFrame, cfg: PanelPipelineConfig) -> dict[str, Any]:
    return VectorizedPanelDataset(
        data=panel,
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        targets=cfg.features.target_column,
        in_steps=cfg.features.sequence_length,
        horizon=cfg.features.forecast_horizon,
        out_steps=1,
        dynamic_features=cfg.features.continuous_columns,
        dynamic_categorical_features=cfg.features.dynamic_categorical_columns or None,
        static_categorical_features=cfg.features.static_categorical_columns or None,
    ).generate_sequences()
