"""
Panel Ensemble Runner
---------------------
End-to-end bagging ensemble trainer for panel forecasting with fast per-
estimator preprocessing.

Design
------
The raw training panel is vectorized **once** at startup:

- A global :class:`VocabularyTokenizer` is fit on the full training panel
  (semantically safe — tokenizer vocabulary is scope-of-universe only, not
  temporal statistics).
- Continuous columns stay raw (unscaled) in the vectorized tensors.
- All rolling forecast windows land in the big ``(B, ID, T, K)`` / target
  arrays keyed by forecast-start-date.

Per estimator the runner then:

1. Samples in-bag forecast dates (``DateBootstrapper``) and OOB dates.
2. Optionally subsamples assets (``AssetSubspaceBootstrapper``).
3. Slices the global vectorized tensors along the B (date) and I (asset)
   axes — pure numpy, no pandas, no Python per-window loop.
4. Fits a per-estimator :class:`StandardScaler` on the long-format in-bag
   slice (cheap groupby mean/std; keeps leak-free temporal statistics).
5. Applies the scaler to the sliced tensors as a broadcast subtract+divide
   across the B and T axes.
6. Trains with OOB-as-val so ``EarlyStopping`` just works.
7. Saves a ``PreprocessingBundle`` (global tokenizer + per-estimator
   scaler) plus model weights + per-estimator manifest; inference is
   unchanged.

Functions
---------
run_panel_ensemble
    Top-level ensemble training entry point called by ``scripts/ensemble.py``.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import dlecosys.models  # noqa: F401 — triggers @register side-effects

from dlecosys.shared.config.panel_schema import PanelPipelineConfig
from dlecosys.shared.data.panel.dataset import VectorizedPanelTorchDataset
from dlecosys.shared.data.panel.vectorizer import VectorizedPanelDataset
from dlecosys.shared.ensembling.panel.bootstrappers import (
    AssetSubspaceBootstrapper,
    BaseFeatureBootstrapper,
    BaseSampleBootstrapper,
    DateBootstrapper,
    NoFeatureBootstrapper,
    NoSampleBootstrapper,
)
from dlecosys.shared.ensembling.panel.display import (
    print_estimator_result,
    render_header,
    render_leaderboard,
)
from dlecosys.shared.ensembling.panel.manifest import (
    PanelEnsembleEstimatorEntry,
    PanelEnsembleManifest,
)
from dlecosys.shared.ensembling.panel.pruning import prune_top_n
from dlecosys.shared.models import ModelFactory
from dlecosys.shared.preprocessing.panel import (
    PreprocessingBundle,
    StandardScaler,
    VocabularyTokenizer,
)
from dlecosys.shared.training.callbacks import EarlyStopping
from dlecosys.shared.training.panel_losses import MaskedGaussianLogLikelihood
from dlecosys.shared.training.panel_manifest import PanelTrainingManifest
from dlecosys.shared.training.panel_trainer import PanelTrainer

__all__ = ["run_panel_ensemble"]

logger = logging.getLogger(__name__)


def run_panel_ensemble(cfg: PanelPipelineConfig) -> Path:
    """Train a bagging ensemble of panel forecasting models."""

    if cfg.ensemble is None:
        raise ValueError("run_panel_ensemble requires cfg.ensemble to be set.")

    torch.manual_seed(cfg.training.random_seed)

    logger.info("Loading training panel for ensemble bagging")
    raw_panel = _load_training_panel(cfg)
    unique_dates = _sorted_unique_dates(raw_panel, cfg.data.date_column)
    unique_assets = list(raw_panel[cfg.data.id_column].drop_duplicates().tolist())

    categorical_columns = (
        cfg.features.dynamic_categorical_columns + cfg.features.static_categorical_columns
    )

    logger.info("Fitting global tokenizer and vectorizing raw training panel (once)")
    global_tokenizer = _fit_global_tokenizer(raw_panel, categorical_columns)
    tokenized_panel = _apply_tokenizer(raw_panel, global_tokenizer, categorical_columns)
    global_vectorized = _vectorize(tokenized_panel, cfg)
    if global_vectorized["y"] is None:
        raise ValueError("Training panel produced no valid forecast windows.")
    forecast_dates = np.asarray(global_vectorized["forecast_start_dates"], dtype="datetime64[ns]")
    global_asset_order: list[Any] = list(global_vectorized["id_index"])

    sample_bs = _build_sample_bootstrapper(cfg)
    feature_bs = _build_feature_bootstrapper(cfg)

    ensemble_dir = (cfg.artifacts.output_path() / "ensemble").resolve()
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    estimator_entries: list[PanelEnsembleEstimatorEntry] = []
    result_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    run_name = cfg.artifacts.output_path().name
    monitor = cfg.early_stopping.monitor if cfg.early_stopping.enabled else "val_loss"
    rng = np.random.default_rng(cfg.ensemble.seed)

    for estimator_index in range(cfg.ensemble.n_estimators):
        estimator_seed = int(rng.integers(0, 2**31 - 1))
        estimator_rng = np.random.default_rng(estimator_seed)

        in_bag_dates, oob_dates = sample_bs.sample(unique_dates, rng=estimator_rng)
        chosen_assets = feature_bs.sample(unique_assets, rng=estimator_rng)
        unique_in_bag = pd.unique(pd.to_datetime(in_bag_dates))
        unique_in_bag_count = int(unique_in_bag.size)

        render_header(
            run_name=run_name,
            estimator_id=estimator_index,
            n_estimators=cfg.ensemble.n_estimators,
            monitor=monitor,
            in_bag_dates=unique_in_bag_count,
            oob_dates=int(len(oob_dates)),
            assets=int(len(chosen_assets)),
            total_assets=len(unique_assets),
            seed=estimator_seed,
            best=best,
            completed=len(result_rows),
        )

        estimator_dir = ensemble_dir / f"estimator_{estimator_index}"
        estimator_dir.mkdir(parents=True, exist_ok=True)

        entry = _train_one_estimator(
            cfg=cfg,
            estimator_index=estimator_index,
            estimator_seed=estimator_seed,
            raw_panel=raw_panel,
            global_vectorized=global_vectorized,
            global_forecast_dates=forecast_dates,
            global_asset_order=global_asset_order,
            global_tokenizer=global_tokenizer,
            in_bag_dates=unique_in_bag,
            oob_dates=oob_dates,
            chosen_assets=chosen_assets,
            estimator_dir=estimator_dir,
        )
        estimator_entries.append(entry)

        oob_value = entry.oob_val_loss
        result_rows.append(
            {
                "estimator_id": estimator_index,
                "oob_value": oob_value,
                "in_bag_dates": unique_in_bag_count,
                "oob_dates": int(len(oob_dates)),
                "assets": int(len(chosen_assets)),
            }
        )
        if oob_value is not None and (best is None or oob_value < best["value"]):
            best = {"id": estimator_index, "value": oob_value}
        print_estimator_result(
            estimator_id=estimator_index, oob_value=oob_value, best=best
        )

    if cfg.ensemble.pruning.enabled:
        oob_losses = [
            entry.oob_val_loss if entry.oob_val_loss is not None else float("inf")
            for entry in estimator_entries
        ]
        keep_indices = prune_top_n(oob_losses=oob_losses, keep=cfg.ensemble.pruning.keep)
        survivors = [estimator_entries[i] for i in keep_indices]
    else:
        survivors = estimator_entries

    manifest = PanelEnsembleManifest(
        ensemble_type=cfg.ensemble.type,
        aggregation=cfg.ensemble.aggregation,
        estimators=survivors,
    )
    manifest_path = ensemble_dir / "ensemble_manifest.json"
    manifest.save(manifest_path)

    render_leaderboard(
        run_name=run_name,
        monitor=monitor,
        aggregation=cfg.ensemble.aggregation,
        n_estimators=cfg.ensemble.n_estimators,
        results=result_rows,
        best=best,
        pruning_enabled=cfg.ensemble.pruning.enabled,
        pruning_strategy=cfg.ensemble.pruning.strategy,
        retained_ids=[entry.index for entry in survivors],
    )
    logger.info("Wrote ensemble manifest with %d survivors to %s", len(survivors), manifest_path)
    return ensemble_dir


def _train_one_estimator(
    *,
    cfg: PanelPipelineConfig,
    estimator_index: int,
    estimator_seed: int,
    raw_panel: pd.DataFrame,
    global_vectorized: dict[str, Any],
    global_forecast_dates: np.ndarray,
    global_asset_order: list[Any],
    global_tokenizer: VocabularyTokenizer | None,
    in_bag_dates: np.ndarray,
    oob_dates: np.ndarray,
    chosen_assets: np.ndarray,
    estimator_dir: Path,
) -> PanelEnsembleEstimatorEntry:
    torch.manual_seed(estimator_seed)

    date_column = cfg.data.date_column
    id_column = cfg.data.id_column
    continuous_columns = cfg.features.continuous_columns
    target_column = cfg.features.target_column

    asset_idx = _asset_indices(global_asset_order=global_asset_order, chosen_assets=chosen_assets)
    in_bag_b_idx = _indices_in(global_forecast_dates, in_bag_dates)
    oob_b_idx = _indices_in(global_forecast_dates, oob_dates) if len(oob_dates) > 0 else np.array([], dtype=int)
    if in_bag_b_idx.size == 0:
        raise ValueError(
            f"Estimator {estimator_index}: bootstrap produced no forecast windows."
        )

    scaler = _fit_scaler_on_slice(
        raw_panel=raw_panel,
        id_column=id_column,
        date_column=date_column,
        continuous_columns=continuous_columns,
        in_bag_dates=in_bag_dates,
        chosen_assets=chosen_assets,
    )

    train_vectorized = _slice_and_scale(
        global_vectorized=global_vectorized,
        b_idx=in_bag_b_idx,
        asset_idx=asset_idx,
        chosen_assets=chosen_assets,
        scaler=scaler,
        continuous_columns=continuous_columns,
        target_column=target_column,
    )
    train_dataset = VectorizedPanelTorchDataset(train_vectorized)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
    )

    val_loader = None
    if oob_b_idx.size > 0:
        oob_vectorized = _slice_and_scale(
            global_vectorized=global_vectorized,
            b_idx=oob_b_idx,
            asset_idx=asset_idx,
            chosen_assets=chosen_assets,
            scaler=scaler,
            continuous_columns=continuous_columns,
            target_column=target_column,
        )
        if oob_vectorized["y"] is not None and oob_vectorized["y"].shape[0] > 0:
            val_dataset = VectorizedPanelTorchDataset(oob_vectorized)
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.training.batch_size,
                shuffle=False,
            )

    preprocessing = PreprocessingBundle(
        continuous_scaler=scaler,
        categorical_tokenizer=global_tokenizer,
        continuous_columns=continuous_columns,
        categorical_columns=(
            cfg.features.dynamic_categorical_columns
            + cfg.features.static_categorical_columns
        ) or None,
        id_column=id_column,
    )

    resolved_params = _resolve_model_params(cfg, preprocessing)
    model = ModelFactory.build(cfg.model.name, resolved_params)
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    callbacks: list[Any] = []
    if val_loader is not None and cfg.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.early_stopping.monitor,
                mode=cfg.early_stopping.mode,
                patience=cfg.early_stopping.patience,
                min_delta=cfg.early_stopping.min_delta,
                warmup=cfg.early_stopping.warmup,
                restore_best_weights=cfg.early_stopping.restore_best_weights,
                verbose=cfg.early_stopping.verbose,
            )
        )

    device = torch.device(cfg.training.device)
    trainer = PanelTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=MaskedGaussianLogLikelihood(),
        metrics={},
        callbacks=callbacks,
        verbose=cfg.training.verbose,
        strict=False,
        device=device,
        grad_clip=cfg.training.grad_clip,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=cfg.training.max_epochs,
    )

    oob_loss = None
    if val_loader is not None:
        final_logs = trainer.logger.last_log()
        oob_loss = float(final_logs.get("val_loss")) if "val_loss" in final_logs else None

    preprocessing_path = (estimator_dir / "preprocessing.json").resolve()
    model_state_path = (estimator_dir / "model.pt").resolve()
    estimator_manifest_path = (estimator_dir / "training_manifest.json").resolve()

    preprocessing.save(preprocessing_path)
    torch.save(model.state_dict(), model_state_path)
    manifest = PanelTrainingManifest(
        preprocessing_path=str(preprocessing_path),
        model_state_path=str(model_state_path),
        model_name=cfg.model.name,
        model_params=resolved_params,
        sequence_length=cfg.features.sequence_length,
        forecast_horizon=cfg.features.forecast_horizon,
        target_column=cfg.features.target_column,
        date_column=cfg.data.date_column,
        id_column=cfg.data.id_column,
        continuous_columns=cfg.features.continuous_columns,
        dynamic_categorical_columns=cfg.features.dynamic_categorical_columns,
        static_categorical_columns=cfg.features.static_categorical_columns,
    )
    manifest.save(estimator_manifest_path)

    return PanelEnsembleEstimatorEntry(
        index=estimator_index,
        estimator_manifest_path=str(estimator_manifest_path),
        oob_val_loss=oob_loss,
    )


def _fit_global_tokenizer(
    raw_panel: pd.DataFrame, categorical_columns: Sequence[str]
) -> VocabularyTokenizer | None:
    """Fit a single tokenizer across the full training panel."""

    if not categorical_columns:
        return None
    return VocabularyTokenizer().fit(raw_panel.loc[:, list(categorical_columns)])


def _apply_tokenizer(
    raw_panel: pd.DataFrame,
    tokenizer: VocabularyTokenizer | None,
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Apply the global tokenizer in place on a copy of ``raw_panel``."""

    if tokenizer is None or not categorical_columns:
        return raw_panel
    out = raw_panel.copy()
    tokenized = tokenizer.transform(out.loc[:, list(categorical_columns)]).astype("int64")
    for column in categorical_columns:
        out[column] = tokenized[column]
    return out


def _fit_scaler_on_slice(
    *,
    raw_panel: pd.DataFrame,
    id_column: str,
    date_column: str,
    continuous_columns: Sequence[str],
    in_bag_dates: np.ndarray,
    chosen_assets: np.ndarray,
) -> StandardScaler | None:
    """Fit ``StandardScaler`` on the in-bag × chosen-assets slice of the raw panel."""

    if not continuous_columns:
        return None
    asset_mask = raw_panel[id_column].isin(chosen_assets)
    date_mask = raw_panel[date_column].isin(in_bag_dates)
    slice_ = raw_panel.loc[asset_mask & date_mask, [id_column, *continuous_columns]]
    if slice_.empty:
        raise ValueError("In-bag slice of raw panel is empty.")
    return StandardScaler(id_column=id_column).fit(slice_)


def _slice_and_scale(
    *,
    global_vectorized: dict[str, Any],
    b_idx: np.ndarray,
    asset_idx: np.ndarray,
    chosen_assets: np.ndarray,
    scaler: StandardScaler | None,
    continuous_columns: Sequence[str],
    target_column: str,
) -> dict[str, Any]:
    """Slice the global vectorized panel and apply the per-estimator scaler."""

    result: dict[str, Any] = {}
    for key in (
        "X_continuous",
        "X_continuous_mask",
        "X_cat_continuous",
        "X_cat_continuous_mask",
        "X_cat_static",
        "X_cat_static_mask",
        "y",
        "y_mask",
    ):
        arr = global_vectorized.get(key)
        if arr is None:
            result[key] = None
        else:
            result[key] = arr[b_idx][:, asset_idx]

    result["id_index"] = list(chosen_assets)
    result["forecast_start_dates"] = global_vectorized["forecast_start_dates"][b_idx]
    result["forecast_end_dates"] = global_vectorized["forecast_end_dates"][b_idx]

    if scaler is not None and result.get("X_continuous") is not None:
        result["X_continuous"] = _apply_tensor_scaling(
            result["X_continuous"],
            scaler=scaler,
            assets=chosen_assets,
            feature_order=continuous_columns,
        )
    if (
        scaler is not None
        and result.get("y") is not None
        and target_column in (continuous_columns or ())
    ):
        result["y"] = _apply_tensor_scaling(
            result["y"],
            scaler=scaler,
            assets=chosen_assets,
            feature_order=(target_column,),
        )
    return result


def _apply_tensor_scaling(
    tensor: np.ndarray,
    *,
    scaler: StandardScaler,
    assets: Sequence[Any],
    feature_order: Sequence[str],
) -> np.ndarray:
    """Apply per-asset per-feature mean/scale as a broadcast over ``(B, I, ?, D)``."""

    mean_arr = np.asarray(
        [[scaler.mean_[asset][feature] for feature in feature_order] for asset in assets],
        dtype=tensor.dtype,
    )
    scale_arr = np.asarray(
        [[scaler.scale_[asset][feature] for feature in feature_order] for asset in assets],
        dtype=tensor.dtype,
    )
    mean_bcast = mean_arr[None, :, None, :]
    scale_bcast = scale_arr[None, :, None, :]
    return (tensor - mean_bcast) / scale_bcast


def _asset_indices(
    *, global_asset_order: Sequence[Any], chosen_assets: Sequence[Any]
) -> np.ndarray:
    """Map ``chosen_assets`` back to their positions in the global I axis."""

    position = {asset: index for index, asset in enumerate(global_asset_order)}
    missing = [asset for asset in chosen_assets if asset not in position]
    if missing:
        raise ValueError(
            f"Chosen assets missing from the global panel: {missing}"
        )
    return np.asarray([position[asset] for asset in chosen_assets], dtype=int)


def _indices_in(all_dates: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Return the positions of ``selected`` inside ``all_dates``."""

    mask = np.isin(all_dates, np.asarray(selected, dtype="datetime64[ns]"))
    return np.nonzero(mask)[0]


def _build_sample_bootstrapper(cfg: PanelPipelineConfig) -> BaseSampleBootstrapper:
    assert cfg.ensemble is not None
    section = cfg.ensemble.sample_bootstrapper
    if section.type == "with_replacement":
        return DateBootstrapper(max_samples=section.max_samples)
    if section.type == "no_bootstrap":
        return NoSampleBootstrapper()
    raise ValueError(f"Unsupported sample_bootstrapper type {section.type!r}.")


def _build_feature_bootstrapper(cfg: PanelPipelineConfig) -> BaseFeatureBootstrapper:
    assert cfg.ensemble is not None
    section = cfg.ensemble.feature_bootstrapper
    if section.type == "random_subspace":
        return AssetSubspaceBootstrapper(max_features=section.max_features)
    if section.type == "all":
        return NoFeatureBootstrapper()
    raise ValueError(f"Unsupported feature_bootstrapper type {section.type!r}.")


def _load_training_panel(cfg: PanelPipelineConfig) -> pd.DataFrame:
    required = {
        cfg.data.date_column,
        cfg.data.id_column,
        cfg.features.target_column,
        *cfg.features.continuous_columns,
        *cfg.features.dynamic_categorical_columns,
        *cfg.features.static_categorical_columns,
    }
    frame = pd.read_csv(cfg.data.train_csv_path())
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Training panel CSV is missing required columns: {', '.join(missing)}"
        )
    frame = frame.copy()
    frame[cfg.data.date_column] = pd.to_datetime(frame[cfg.data.date_column])
    return frame.sort_values([cfg.data.id_column, cfg.data.date_column]).reset_index(drop=True)


def _sorted_unique_dates(panel: pd.DataFrame, date_column: str) -> np.ndarray:
    values = pd.to_datetime(panel[date_column]).dropna().unique()
    if len(values) == 0:
        raise ValueError("Training panel contains no valid dates for bootstrapping.")
    return np.sort(values)


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


def _resolve_model_params(
    cfg: PanelPipelineConfig, preprocessing: PreprocessingBundle
) -> dict[str, Any]:
    params: dict[str, Any] = copy.deepcopy(dict(cfg.model.params))
    params.setdefault(
        "continuous_dim",
        len(cfg.features.continuous_columns) if cfg.features.continuous_columns else None,
    )
    tokenizer = preprocessing.categorical_tokenizer
    dynamic_cols = cfg.features.dynamic_categorical_columns
    static_cols = cfg.features.static_categorical_columns
    if (dynamic_cols or static_cols) and tokenizer is None:
        raise ValueError("Categorical streams require a fitted tokenizer.")
    if dynamic_cols:
        params["dynamic_categorical_cardinalities"] = list(
            tokenizer.get_vocabulary_sizes(dynamic_cols)
        )
    if static_cols:
        params["static_categorical_cardinalities"] = list(
            tokenizer.get_vocabulary_sizes(static_cols)
        )
    return params
