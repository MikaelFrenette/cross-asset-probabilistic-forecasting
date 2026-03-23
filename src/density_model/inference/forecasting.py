"""
Forecast Inference
------------------
Prediction-time utilities for validating external panel CSV inputs, aligning
them to the configured master market calendar, and producing rolling Gaussian
forecasts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from density_model.config import CsvDatasetConfig, PredictConfig, TrainingArtifactManifest, load_training_manifest
from density_model.data import XNYSCalendar
from density_model.datasets import VectorizedPanelDataset
from density_model.preprocessing import BasePreprocessorArtifact, PreprocessingBundle

__all__ = [
    "build_online_prediction_payload",
    "build_rolling_prediction_panel_data",
    "predict_from_config",
]


def predict_from_config(
    *,
    config: PredictConfig,
) -> Path:
    """
    Run next-session inference from a prepared external panel CSV file.

    Parameters
    ----------
    config : PredictConfig
        Typed CSV-based prediction configuration.

    Returns
    -------
    pathlib.Path
        Output CSV path containing rolling forecast means and standard
        deviations.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only when torch is unavailable.
        raise ImportError("torch must be installed to run the prediction workflow.") from error

    from density_model.models import PanelBatch, build_panel_model

    manifest = load_training_manifest(config.manifest_file())
    artifact = BasePreprocessorArtifact.load(manifest.preprocessing_artifact_path())
    if not isinstance(artifact, PreprocessingBundle):
        raise TypeError("The configured preprocessing artifact must deserialize to PreprocessingBundle.")
    preprocessing = artifact
    model = build_panel_model(
        model_config=manifest.model_config_runtime,
        feature_config=manifest.feature_config,
        preprocessing=preprocessing,
    )
    state_dict = torch.load(manifest.model_weights_path(), map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    raw_panel = _load_prediction_panel(path=config.dataset.csv_file(), manifest=manifest, config=config)
    panel_data = build_rolling_prediction_panel_data(
        panel=raw_panel,
        manifest=manifest,
        preprocessing=preprocessing,
        dataset_config=config.dataset,
    )
    if panel_data["y"] is None:
        raise ValueError("Prediction dataset did not produce any valid forecast windows.")
    batch_size = 256
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start_index in range(0, int(panel_data["y"].shape[0]), batch_size):
            end_index = min(start_index + batch_size, int(panel_data["y"].shape[0]))
            features = {
                key: _to_torch_tensor(value[start_index:end_index]).to(config.device)
                for key, value in panel_data.items()
                if key.startswith("X_") and value is not None
            }
            batch = PanelBatch.from_training_batch(
                features=features,
                y=torch.zeros(
                    (end_index - start_index, len(panel_data["id_index"]), 1, 1),
                    dtype=torch.float32,
                    device=config.device,
                ),
                y_mask=torch.ones(
                    (end_index - start_index, len(panel_data["id_index"]), 1, 1),
                    dtype=torch.bool,
                    device=config.device,
                ),
                id_index=list(panel_data["id_index"]),
                forecast_start_dates=list(pd.to_datetime(panel_data["forecast_start_dates"][start_index:end_index])),
                forecast_end_dates=list(pd.to_datetime(panel_data["forecast_end_dates"][start_index:end_index])),
            )
            prediction = model(batch).cpu()
            mu, sigma = torch.chunk(prediction, chunks=2, dim=-1)
            mu_output, sigma_output = _inverse_transform_return_distribution(
                mu=mu,
                sigma=sigma,
                id_index=list(panel_data["id_index"]),
                preprocessing=preprocessing,
                target_column=manifest.feature_config.target_column,
            )
            forecast_start_dates = pd.to_datetime(panel_data["forecast_start_dates"][start_index:end_index])
            forecast_end_dates = pd.to_datetime(panel_data["forecast_end_dates"][start_index:end_index])
            for batch_offset, forecast_start_date in enumerate(forecast_start_dates):
                for asset_offset, asset_id in enumerate(panel_data["id_index"]):
                    rows.append(
                        {
                            "forecast_start_date": pd.Timestamp(forecast_start_date),
                            "forecast_end_date": pd.Timestamp(forecast_end_dates[batch_offset]),
                            "asset_id": asset_id,
                            "mu": float(mu_output[batch_offset, asset_offset]),
                            "sigma": float(sigma_output[batch_offset, asset_offset]),
                        }
                    )

    output_frame = pd.DataFrame(rows)
    resolved_output_path = config.output_path()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(resolved_output_path, index=False)
    return resolved_output_path


def build_online_prediction_payload(
    *,
    panel: pd.DataFrame,
    manifest: TrainingArtifactManifest,
    preprocessing: PreprocessingBundle,
    dataset_config: CsvDatasetConfig,
) -> dict[str, Any]:
    """
    Build one next-session model payload from the latest available panel history.

    Parameters
    ----------
    panel : pandas.DataFrame
        Raw prediction-time panel with the same public geometry as training
        inputs.
    manifest : TrainingArtifactManifest
        Frozen training manifest.
    preprocessing : PreprocessingBundle
        Fitted preprocessing bundle from training.
    dataset_config : Any
        Prediction dataset schema configuration.

    Returns
    -------
    dict of str to Any
        Payload containing batched feature tensors, identifier order, and the
        next forecast date metadata.
    """

    normalized_panel = _normalize_prediction_panel_to_calendar(
        panel=panel,
        manifest=manifest,
        dataset_config=dataset_config,
    )
    transformed_panel = preprocessing.transform(normalized_panel)

    continuous_columns = manifest.feature_config.streams.resolve_continuous_columns()
    dynamic_categorical_columns = manifest.feature_config.streams.resolve_dynamic_categorical_columns()
    static_categorical_columns = manifest.feature_config.streams.resolve_static_categorical_columns()
    id_order = list(dict.fromkeys(transformed_panel[dataset_config.id_column].tolist()))
    feature_blocks: dict[str, Any] = {
        "X_continuous": [],
        "X_continuous_mask": [],
    }
    if dynamic_categorical_columns:
        feature_blocks["X_cat_continuous"] = []
        feature_blocks["X_cat_continuous_mask"] = []
    if static_categorical_columns:
        feature_blocks["X_cat_static"] = []
        feature_blocks["X_cat_static_mask"] = []

    for entity_id in id_order:
        group = (
            transformed_panel.loc[transformed_panel[dataset_config.id_column] == entity_id]
            .sort_values(dataset_config.date_column)
            .reset_index(drop=True)
        )
        if len(group) < manifest.feature_config.sequence_length:
            raise ValueError(
                f"Prediction dataset does not contain enough history for asset {entity_id!r} "
                f"to build a {manifest.feature_config.sequence_length}-step input window."
            )
        window = group.iloc[-manifest.feature_config.sequence_length :].copy()
        feature_blocks["X_continuous"].append(window.loc[:, list(continuous_columns)].to_numpy(copy=True))
        feature_blocks["X_continuous_mask"].append(_observed_mask(window.loc[:, list(continuous_columns)].to_numpy(copy=True)))

        if dynamic_categorical_columns:
            dynamic_values = window.loc[:, list(dynamic_categorical_columns)].to_numpy(copy=True)
            feature_blocks["X_cat_continuous"].append(dynamic_values)
            feature_blocks["X_cat_continuous_mask"].append(_observed_mask(dynamic_values))

        if static_categorical_columns:
            static_values = window.loc[:, list(static_categorical_columns)].iloc[-1].to_numpy(copy=True)
            static_values = np.broadcast_to(
                static_values[None, :],
                (manifest.feature_config.sequence_length, static_values.shape[0]),
            ).copy()
            feature_blocks["X_cat_static"].append(static_values)
            feature_blocks["X_cat_static_mask"].append(_observed_mask(static_values))

    latest_observed_date = pd.Timestamp(transformed_panel[dataset_config.date_column].max()).normalize()
    calendar = XNYSCalendar(calendar_name=manifest.calendar)
    forecast_start_date = _next_session_after(calendar=calendar, observed_date=latest_observed_date)
    payload = {
        "features": {
            key: _to_torch_tensor(np.stack(values, axis=0))[None, ...] if values else None
            for key, values in feature_blocks.items()
        },
        "id_index": id_order,
        "forecast_start_date": forecast_start_date,
        "forecast_end_date": forecast_start_date,
    }
    return payload


def build_rolling_prediction_panel_data(
    *,
    panel: pd.DataFrame,
    manifest: TrainingArtifactManifest,
    preprocessing: PreprocessingBundle,
    dataset_config: CsvDatasetConfig,
) -> dict[str, Any]:
    """
    Build rolling forecast-aligned panel tensors from a prediction dataset.

    Parameters
    ----------
    panel : pandas.DataFrame
        Raw prediction-time panel with the same public geometry as training
        inputs.
    manifest : TrainingArtifactManifest
        Frozen training manifest.
    preprocessing : PreprocessingBundle
        Fitted preprocessing bundle from training.
    dataset_config : CsvDatasetConfig
        Prediction dataset schema configuration.

    Returns
    -------
    dict of str to Any
        Vectorized panel output aligned to forecast dates.
    """

    normalized_panel = _normalize_prediction_panel_to_calendar(
        panel=panel,
        manifest=manifest,
        dataset_config=dataset_config,
    )
    transformed_panel = preprocessing.transform(normalized_panel)
    return VectorizedPanelDataset(
        data=transformed_panel,
        date_column=dataset_config.date_column,
        id_column=dataset_config.id_column,
        targets=manifest.feature_config.target_column,
        in_steps=manifest.feature_config.sequence_length,
        horizon=manifest.feature_config.forecast_horizon,
        out_steps=1,
        dynamic_features=manifest.feature_config.streams.resolve_continuous_columns(),
        dynamic_categorical_features=manifest.feature_config.streams.resolve_dynamic_categorical_columns() or None,
        static_categorical_features=manifest.feature_config.streams.resolve_static_categorical_columns() or None,
    ).generate_sequences()


def _load_prediction_panel(
    *,
    path: str | Path,
    manifest: TrainingArtifactManifest,
    config: PredictConfig,
) -> pd.DataFrame:
    """
    Load and validate an external prediction-time panel CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Input CSV path.
    manifest : TrainingArtifactManifest
        Frozen training manifest.
    config : PredictConfig
        Typed prediction configuration containing dataset schema.

    Returns
    -------
    pandas.DataFrame
        Canonical raw prediction panel sorted by identifier and date.
    """

    frame = pd.read_csv(Path(path))
    date_column = config.dataset.date_column
    id_column = config.dataset.id_column
    required_columns = [
        date_column,
        id_column,
        manifest.feature_config.target_column,
        *manifest.feature_config.streams.resolve_continuous_columns(),
        *manifest.feature_config.streams.resolve_dynamic_categorical_columns(),
        *manifest.feature_config.streams.resolve_static_categorical_columns(),
    ]
    ordered_required_columns = list(dict.fromkeys(required_columns))
    missing_columns = sorted(set(ordered_required_columns).difference(frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Prediction dataset is missing required columns: {missing_text}")
    frame = frame.loc[:, ordered_required_columns].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)


def _normalize_prediction_panel_to_calendar(
    *,
    panel: pd.DataFrame,
    manifest: TrainingArtifactManifest,
    dataset_config: CsvDatasetConfig,
) -> pd.DataFrame:
    """
    Align a raw prediction-time panel to the master XNYS calendar.

    Parameters
    ----------
    panel : pandas.DataFrame
        Raw prediction panel.
    manifest : TrainingArtifactManifest
        Frozen training manifest.
    dataset_config : Any
        Prediction dataset schema configuration.

    Returns
    -------
    pandas.DataFrame
        Calendar-normalized panel preserving missing sessions as missing values.
    """

    if panel.empty:
        raise ValueError("Prediction dataset is empty.")
    calendar = XNYSCalendar(calendar_name=manifest.calendar)
    start_date = pd.Timestamp(panel[dataset_config.date_column].min()).date()
    end_date = pd.Timestamp(panel[dataset_config.date_column].max()).date()
    sessions = calendar.sessions_in_range(start_date=start_date, end_date=end_date)
    normalized_frames: list[pd.DataFrame] = []
    static_categorical_columns = manifest.feature_config.streams.resolve_static_categorical_columns()

    for entity_id, group in panel.groupby(dataset_config.id_column, sort=False):
        aligned = group.set_index(dataset_config.date_column).reindex(sessions).reset_index(names=dataset_config.date_column)
        aligned[dataset_config.id_column] = entity_id
        for column in static_categorical_columns:
            non_missing = group[column].dropna().unique().tolist()
            if len(non_missing) > 1:
                raise ValueError(f"Static categorical column {column!r} is not constant within asset {entity_id!r}.")
            aligned[column] = non_missing[0] if non_missing else None
        normalized_frames.append(aligned)

    normalized = pd.concat(normalized_frames, ignore_index=True)
    return normalized.sort_values([dataset_config.id_column, dataset_config.date_column]).reset_index(drop=True)


def _next_session_after(*, calendar: XNYSCalendar, observed_date: pd.Timestamp) -> pd.Timestamp:
    """
    Resolve the next XNYS session strictly after the last observed date.

    Parameters
    ----------
    calendar : XNYSCalendar
        XNYS session helper.
    observed_date : pandas.Timestamp
        Last observed date in the prediction panel.

    Returns
    -------
    pandas.Timestamp
        Next XNYS session date.
    """

    sessions = calendar.sessions_in_range(
        start_date=observed_date.date(),
        end_date=(observed_date + pd.Timedelta(days=14)).date(),
    )
    future_sessions = sessions[sessions > observed_date]
    if future_sessions.empty:
        raise ValueError("Unable to resolve the next XNYS session after the latest observed date.")
    return pd.Timestamp(future_sessions[0]).normalize()


def _observed_mask(array: np.ndarray) -> np.ndarray:
    """
    Build a feature-level observation mask for a NumPy array.

    Parameters
    ----------
    array : numpy.ndarray
        Feature array.

    Returns
    -------
    numpy.ndarray
        Boolean observation mask with the same shape as ``array``.
    """

    if np.issubdtype(array.dtype, np.floating):
        return ~np.isnan(array)
    return np.ones(array.shape, dtype=bool)


def _inverse_transform_return_distribution(
    *,
    mu: Any,
    sigma: Any,
    id_index: list[Any],
    preprocessing: PreprocessingBundle,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map one-step Gaussian outputs from scaled return space back to return space.

    Parameters
    ----------
    mu : Any
        Predicted mean tensor in scaled return space.
    sigma : Any
        Predicted standard deviation tensor in scaled return space.
    id_index : list of Any
        Ordered asset identifiers aligned with the ``ID`` axis.
    preprocessing : PreprocessingBundle
        Fitted preprocessing bundle containing the per-asset return scaler.
    target_column : str
        Supervised target column name.

    Returns
    -------
    tuple of numpy.ndarray and numpy.ndarray
        Mean and standard deviation arrays in return space.
    """

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


def _to_torch_tensor(array: np.ndarray) -> Any:
    """
    Convert one NumPy feature block to a torch tensor.

    Parameters
    ----------
    array : numpy.ndarray
        Feature block array.

    Returns
    -------
    Any
        Torch tensor with the appropriate dtype.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only when torch is unavailable.
        raise ImportError("torch must be installed to prepare inference tensors.") from error

    if np.issubdtype(array.dtype, np.bool_):
        return torch.as_tensor(array, dtype=torch.bool)
    if np.issubdtype(array.dtype, np.integer):
        return torch.as_tensor(array, dtype=torch.long)
    return torch.as_tensor(array, dtype=torch.float32)
