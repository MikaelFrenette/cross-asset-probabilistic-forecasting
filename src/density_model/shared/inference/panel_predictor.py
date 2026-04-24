"""
Panel Predictor
---------------
Inference utilities for panel forecasting models. Routes between single-model
and ensemble inference based on the supplied manifest shape:

- Single-model: ``manifest_path`` deserializes as :class:`PanelTrainingManifest`.
  Load the preprocessing bundle + model weights, calendar-normalize the raw
  prediction CSV, vectorize rolling forecast windows, run batched inference,
  and inverse-transform ``(mu, sigma)`` back to return space.

- Ensemble: ``manifest_path`` deserializes as :class:`PanelEnsembleManifest`.
  Iterate the surviving estimators, run per-estimator inference (each with its
  own preprocessing bundle + weights), then aggregate Gaussians via
  ``aggregate_gaussian_predictions`` (mean/median for mu, law-of-total-variance
  for sigma) before writing the unified predictions CSV.

Functions
---------
predict_from_config
    Top-level entry point for ``scripts/predict.py``.
build_rolling_prediction_panel_data
    Lower-level helper that produces the vectorized rolling forecast dict.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import density_model.models  # noqa: F401 — ensures @register side-effects fire

from density_model.shared.config.panel_schema import PanelPredictConfig, PanelPredictDataConfig
from density_model.shared.data.panel.calendar import XNYSCalendar
from density_model.shared.data.panel.schema import PanelBatch
from density_model.shared.data.panel.vectorizer import VectorizedPanelDataset
from density_model.shared.ensembling.panel.aggregation import aggregate_gaussian_predictions
from density_model.shared.ensembling.panel.manifest import load_panel_ensemble_manifest
from density_model.shared.models import ModelFactory
from density_model.shared.preprocessing.panel import BasePreprocessorArtifact, PreprocessingBundle
from density_model.shared.training.panel_manifest import PanelTrainingManifest, load_panel_manifest

__all__ = ["build_rolling_prediction_panel_data", "predict_from_config"]

logger = logging.getLogger(__name__)


def predict_from_config(cfg: PanelPredictConfig) -> Path:
    """Route inference to the single-model or ensemble path based on the manifest."""

    raw = json.loads(cfg.manifest_file().read_text(encoding="utf-8"))
    if "ensemble_type" in raw:
        return _predict_ensemble(cfg)
    return _predict_single(cfg)


def _predict_single(cfg: PanelPredictConfig) -> Path:
    manifest = load_panel_manifest(cfg.manifest_file())
    preprocessing = _load_preprocessing_bundle(manifest.preprocessing_artifact_path())
    model = _build_and_load_model(manifest=manifest, device_str=cfg.device)
    raw_panel = _load_prediction_panel(
        path=cfg.data.csv_file(),
        manifest=manifest,
        data_cfg=cfg.data,
    )
    panel_data = build_rolling_prediction_panel_data(
        panel=raw_panel,
        manifest=manifest,
        preprocessing=preprocessing,
        data_cfg=cfg.data,
    )
    if panel_data["y"] is None:
        raise ValueError("Prediction dataset did not produce any valid forecast windows.")

    mu_stack, sigma_stack = _run_model_inference(
        model=model,
        panel_data=panel_data,
        device=torch.device(cfg.device),
        batch_size=cfg.batch_size,
        preprocessing=preprocessing,
        target_column=manifest.target_column,
    )
    rows = _format_prediction_rows(
        panel_data=panel_data,
        mu=mu_stack,
        sigma=sigma_stack,
    )
    return _write_predictions(cfg, rows)


def _predict_ensemble(cfg: PanelPredictConfig) -> Path:
    ensemble_manifest = load_panel_ensemble_manifest(cfg.manifest_file())
    if not ensemble_manifest.estimators:
        raise ValueError("Ensemble manifest contains no surviving estimators.")

    raw_csv = cfg.data.csv_file()
    device = torch.device(cfg.device)

    per_estimator_mus: list[np.ndarray] = []
    per_estimator_sigmas: list[np.ndarray] = []
    reference_panel_data: dict[str, Any] | None = None
    reference_manifest: PanelTrainingManifest | None = None

    for entry in ensemble_manifest.estimators:
        manifest = load_panel_manifest(Path(entry.estimator_manifest_path))
        preprocessing = _load_preprocessing_bundle(manifest.preprocessing_artifact_path())
        model = _build_and_load_model(manifest=manifest, device_str=cfg.device)

        raw_panel = _load_prediction_panel(
            path=raw_csv, manifest=manifest, data_cfg=cfg.data
        )
        panel_data = build_rolling_prediction_panel_data(
            panel=raw_panel,
            manifest=manifest,
            preprocessing=preprocessing,
            data_cfg=cfg.data,
        )
        if panel_data["y"] is None:
            raise ValueError(
                f"Estimator {entry.index} produced no valid forecast windows on the prediction CSV."
            )

        mu, sigma = _run_model_inference(
            model=model,
            panel_data=panel_data,
            device=device,
            batch_size=cfg.batch_size,
            preprocessing=preprocessing,
            target_column=manifest.target_column,
        )
        per_estimator_mus.append(mu)
        per_estimator_sigmas.append(sigma)
        if reference_panel_data is None:
            reference_panel_data = panel_data
            reference_manifest = manifest
        else:
            _check_aligned(panel_data=panel_data, reference=reference_panel_data)

    assert reference_panel_data is not None and reference_manifest is not None

    mu_stack = np.stack(per_estimator_mus, axis=0)
    sigma_stack = np.stack(per_estimator_sigmas, axis=0)
    mu_ens, sigma_ens = aggregate_gaussian_predictions(
        mus=mu_stack, sigmas=sigma_stack, mode=ensemble_manifest.aggregation
    )
    rows = _format_prediction_rows(
        panel_data=reference_panel_data,
        mu=mu_ens,
        sigma=sigma_ens,
    )
    logger.info(
        "Aggregated %d estimators (mode=%s) into %d prediction rows",
        len(ensemble_manifest.estimators),
        ensemble_manifest.aggregation,
        len(rows),
    )
    return _write_predictions(cfg, rows)


def build_rolling_prediction_panel_data(
    *,
    panel: pd.DataFrame,
    manifest: PanelTrainingManifest,
    preprocessing: PreprocessingBundle,
    data_cfg: PanelPredictDataConfig,
) -> dict[str, Any]:
    """Produce vectorized rolling forecast arrays from a raw prediction panel."""

    normalized_panel = _normalize_prediction_panel_to_calendar(
        panel=panel, manifest=manifest, data_cfg=data_cfg
    )
    transformed_panel = preprocessing.transform(normalized_panel)
    return VectorizedPanelDataset(
        data=transformed_panel,
        date_column=data_cfg.date_column,
        id_column=data_cfg.id_column,
        targets=manifest.target_column,
        in_steps=manifest.sequence_length,
        horizon=manifest.forecast_horizon,
        out_steps=1,
        dynamic_features=manifest.continuous_columns,
        dynamic_categorical_features=manifest.dynamic_categorical_columns or None,
        static_categorical_features=manifest.static_categorical_columns or None,
    ).generate_sequences()


def _build_and_load_model(
    *, manifest: PanelTrainingManifest, device_str: str
) -> torch.nn.Module:
    model = ModelFactory.build(manifest.model_name, manifest.model_params)
    state_dict = torch.load(manifest.model_weights_path(), map_location=device_str)
    model.load_state_dict(state_dict)
    model.to(torch.device(device_str))
    model.eval()
    return model


def _load_preprocessing_bundle(path: Path) -> PreprocessingBundle:
    artifact = BasePreprocessorArtifact.load(path)
    if not isinstance(artifact, PreprocessingBundle):
        raise TypeError(
            f"Preprocessing artifact at {path} must deserialize to PreprocessingBundle."
        )
    return artifact


def _run_model_inference(
    *,
    model: torch.nn.Module,
    panel_data: dict[str, Any],
    device: torch.device,
    batch_size: int,
    preprocessing: PreprocessingBundle,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batched rolling inference and return ``(mu, sigma)`` in return space."""

    total = int(panel_data["y"].shape[0])
    id_index = list(panel_data["id_index"])
    mu_accumulator = np.zeros((total, len(id_index)), dtype=float)
    sigma_accumulator = np.zeros((total, len(id_index)), dtype=float)

    with torch.inference_mode():
        for start_index in range(0, total, batch_size):
            end_index = min(start_index + batch_size, total)
            features = {
                key: _to_torch_tensor(value[start_index:end_index]).to(device)
                for key, value in panel_data.items()
                if key.startswith("X_") and value is not None
            }
            forecast_start_dates = pd.to_datetime(
                panel_data["forecast_start_dates"][start_index:end_index]
            )
            forecast_end_dates = pd.to_datetime(
                panel_data["forecast_end_dates"][start_index:end_index]
            )
            batch = PanelBatch.from_training_batch(
                features=features,
                y=torch.zeros(
                    (end_index - start_index, len(id_index), 1, 1),
                    dtype=torch.float32,
                    device=device,
                ),
                y_mask=torch.ones(
                    (end_index - start_index, len(id_index), 1, 1),
                    dtype=torch.bool,
                    device=device,
                ),
                id_index=id_index,
                forecast_start_dates=list(forecast_start_dates),
                forecast_end_dates=list(forecast_end_dates),
            )
            prediction = model(batch).cpu()
            mu, sigma = torch.chunk(prediction, chunks=2, dim=-1)
            mu_block, sigma_block = _inverse_transform_return_distribution(
                mu=mu,
                sigma=sigma,
                id_index=id_index,
                preprocessing=preprocessing,
                target_column=target_column,
            )
            mu_accumulator[start_index:end_index] = mu_block
            sigma_accumulator[start_index:end_index] = sigma_block

    return mu_accumulator, sigma_accumulator


def _format_prediction_rows(
    *,
    panel_data: dict[str, Any],
    mu: np.ndarray,
    sigma: np.ndarray,
) -> list[dict[str, Any]]:
    forecast_start_dates = pd.to_datetime(panel_data["forecast_start_dates"])
    forecast_end_dates = pd.to_datetime(panel_data["forecast_end_dates"])
    id_index = list(panel_data["id_index"])
    rows: list[dict[str, Any]] = []
    for batch_offset, forecast_start_date in enumerate(forecast_start_dates):
        for asset_offset, asset_id in enumerate(id_index):
            rows.append(
                {
                    "forecast_start_date": pd.Timestamp(forecast_start_date),
                    "forecast_end_date": pd.Timestamp(forecast_end_dates[batch_offset]),
                    "asset_id": asset_id,
                    "mu": float(mu[batch_offset, asset_offset]),
                    "sigma": float(sigma[batch_offset, asset_offset]),
                }
            )
    return rows


def _write_predictions(cfg: PanelPredictConfig, rows: list[dict[str, Any]]) -> Path:
    output_frame = pd.DataFrame(rows)
    resolved_output_path = cfg.output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(resolved_output_path, index=False)
    logger.info("Wrote %d rolling predictions to %s", len(rows), resolved_output_path)
    return resolved_output_path


def _check_aligned(
    *, panel_data: dict[str, Any], reference: dict[str, Any]
) -> None:
    """Ensure subsequent estimators produced the same forecast grid as the reference."""

    if list(panel_data["id_index"]) != list(reference["id_index"]):
        raise ValueError(
            "Ensemble estimators produced misaligned id_index — preprocessing "
            "bundles disagree on asset ordering."
        )
    if panel_data["y"].shape != reference["y"].shape:
        raise ValueError(
            "Ensemble estimators produced misaligned forecast grids — check that "
            "each estimator's feature schema matches the shared prediction CSV."
        )


def _load_prediction_panel(
    *,
    path: Path,
    manifest: PanelTrainingManifest,
    data_cfg: PanelPredictDataConfig,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = data_cfg.date_column
    id_column = data_cfg.id_column
    required_columns = [
        date_column,
        id_column,
        manifest.target_column,
        *manifest.continuous_columns,
        *manifest.dynamic_categorical_columns,
        *manifest.static_categorical_columns,
    ]
    ordered_required = list(dict.fromkeys(required_columns))
    missing_columns = sorted(set(ordered_required).difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Prediction dataset is missing required columns: {', '.join(missing_columns)}"
        )
    frame = frame.loc[:, ordered_required].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)


def _normalize_prediction_panel_to_calendar(
    *,
    panel: pd.DataFrame,
    manifest: PanelTrainingManifest,
    data_cfg: PanelPredictDataConfig,
) -> pd.DataFrame:
    if panel.empty:
        raise ValueError("Prediction dataset is empty.")
    calendar = XNYSCalendar(calendar_name=manifest.calendar)
    start_date = pd.Timestamp(panel[data_cfg.date_column].min()).date()
    end_date = pd.Timestamp(panel[data_cfg.date_column].max()).date()
    sessions = calendar.sessions_in_range(start_date=start_date, end_date=end_date)
    normalized_frames: list[pd.DataFrame] = []
    static_columns = manifest.static_categorical_columns

    for entity_id, group in panel.groupby(data_cfg.id_column, sort=False):
        aligned = (
            group.set_index(data_cfg.date_column)
            .reindex(sessions)
            .reset_index(names=data_cfg.date_column)
        )
        aligned[data_cfg.id_column] = entity_id
        for column in static_columns:
            non_missing = group[column].dropna().unique().tolist()
            if len(non_missing) > 1:
                raise ValueError(
                    f"Static categorical column {column!r} is not constant within "
                    f"asset {entity_id!r}."
                )
            aligned[column] = non_missing[0] if non_missing else None
        normalized_frames.append(aligned)

    normalized = pd.concat(normalized_frames, ignore_index=True)
    return normalized.sort_values([data_cfg.id_column, data_cfg.date_column]).reset_index(
        drop=True
    )


def _inverse_transform_return_distribution(
    *,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    id_index: list[Any],
    preprocessing: PreprocessingBundle,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = preprocessing.continuous_scaler
    if scaler is None or scaler.mean_ is None or scaler.scale_ is None:
        return mu[..., 0, 0].numpy(), sigma[..., 0, 0].numpy()
    if target_column not in (preprocessing.continuous_columns or ()):
        return mu[..., 0, 0].numpy(), sigma[..., 0, 0].numpy()
    mu_values = np.zeros((mu.shape[0], len(id_index)), dtype=float)
    sigma_values = np.zeros((sigma.shape[0], len(id_index)), dtype=float)
    for asset_offset, asset_id in enumerate(id_index):
        mean = float(scaler.mean_[asset_id][target_column])
        scale = float(scaler.scale_[asset_id][target_column])
        mu_values[:, asset_offset] = mu[:, asset_offset, 0, 0].numpy() * scale + mean
        sigma_values[:, asset_offset] = sigma[:, asset_offset, 0, 0].numpy() * scale
    return mu_values, sigma_values


def _to_torch_tensor(array: np.ndarray) -> torch.Tensor:
    if np.issubdtype(array.dtype, np.bool_):
        return torch.as_tensor(array, dtype=torch.bool)
    if np.issubdtype(array.dtype, np.integer):
        return torch.as_tensor(array, dtype=torch.long)
    return torch.as_tensor(array, dtype=torch.float32)
