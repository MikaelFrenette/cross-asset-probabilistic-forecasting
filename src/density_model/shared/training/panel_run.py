"""
Panel Training Runner
---------------------
Training orchestration for panel forecasting. Loads the vectorized train and
validation panels + fitted preprocessing bundle produced by
``run_panel_preprocess`` from disk, derives runtime model parameters from the
fitted bundle, builds the model via :class:`ModelFactory`, runs
:class:`PanelTrainer` (or :class:`PanelDistributedTrainer` under DDP), and
saves the trained model state + training manifest.

DDP mode
--------
When ``cfg.distributed.enabled`` is true, the script must be launched with
``torchrun --nproc_per_node=N``. The runner wraps the training DataLoader
with a ``DistributedSampler``, instantiates ``PanelDistributedTrainer``
(which wraps the model in ``DistributedDataParallel``), sets up the process
group on entry, and tears it down on exit. Artifact writes (model weights
and manifest) are gated to rank 0.

Functions
---------
run_panel_training
    Top-level training entry point called by ``scripts/train.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler

import density_model.models  # noqa: F401 — triggers @register side-effects

from density_model.shared.config.panel_schema import PanelPipelineConfig
from density_model.shared.data.panel.dataset import VectorizedPanelTorchDataset
from density_model.shared.models import ModelFactory
from density_model.shared.preprocessing.panel import BasePreprocessorArtifact, PreprocessingBundle
from density_model.shared.training.callbacks import EarlyStopping, ModelCheckpoint
from density_model.shared.training.panel_losses import MaskedGaussianLogLikelihood
from density_model.shared.training.panel_manifest import PanelTrainingManifest
from density_model.shared.training.panel_preprocess import train_panel_path, val_panel_path
from density_model.shared.training.panel_trainer import PanelTrainer

__all__ = ["run_panel_training"]

logger = logging.getLogger(__name__)


def run_panel_training(cfg: PanelPipelineConfig) -> Path:
    """
    Train a panel forecasting model from preprocessed artifacts.

    Branches on ``cfg.distributed.enabled`` between a single-device
    :class:`PanelTrainer` and a DDP :class:`PanelDistributedTrainer`. Under
    DDP the process group is set up before training and torn down in a
    ``finally``; model weights and manifest are written by rank 0 only.
    """

    if cfg.distributed.enabled:
        return _run_distributed(cfg)
    return _run_single_device(cfg)


def _run_single_device(cfg: PanelPipelineConfig) -> Path:
    torch.manual_seed(cfg.training.random_seed)

    preprocessing = _load_preprocessing_bundle(cfg.artifacts.preprocessing_path())
    train_vectorized = _load_vectorized_panel(train_panel_path(cfg))
    val_vectorized = _load_vectorized_panel(val_panel_path(cfg))

    train_dataset = VectorizedPanelTorchDataset(train_vectorized)
    val_dataset = VectorizedPanelTorchDataset(val_vectorized)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    resolved_params = _resolve_model_params(cfg=cfg, preprocessing=preprocessing)
    logger.info("Building model %r with resolved params", cfg.model.name)
    model = ModelFactory.build(cfg.model.name, resolved_params)
    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    callbacks = _build_callbacks(cfg)
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
    logger.info("Starting single-device training for up to %d epochs", cfg.training.max_epochs)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=cfg.training.max_epochs,
    )

    _save_training_artifacts(cfg=cfg, model=model, resolved_params=resolved_params)
    return cfg.artifacts.output_path().resolve()


def _run_distributed(cfg: PanelPipelineConfig) -> Path:
    from density_model.shared.training.panel_distributed import PanelDistributedTrainer
    from density_model.shared.training.process_group import is_main_process, setup, teardown

    torch.manual_seed(cfg.training.random_seed)

    preprocessing = _load_preprocessing_bundle(cfg.artifacts.preprocessing_path())
    train_vectorized = _load_vectorized_panel(train_panel_path(cfg))
    val_vectorized = _load_vectorized_panel(val_panel_path(cfg))
    train_dataset = VectorizedPanelTorchDataset(train_vectorized)
    val_dataset = VectorizedPanelTorchDataset(val_vectorized)

    setup(cfg.distributed.backend)
    try:
        sampler = DistributedSampler(train_dataset, shuffle=cfg.training.shuffle)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size,
            sampler=sampler,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
        )

        resolved_params = _resolve_model_params(cfg=cfg, preprocessing=preprocessing)
        logger.info("Building model %r with resolved params", cfg.model.name)
        model = ModelFactory.build(cfg.model.name, resolved_params)
        optimizer = Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        # Only rank 0 gets callbacks that write state (checkpoint, early stop
        # restore). Non-rank-0 processes could set stop_training out of sync.
        callbacks = _build_callbacks(cfg) if is_main_process() else []

        trainer = PanelDistributedTrainer(
            train_sampler=sampler,
            model=model,
            optimizer=optimizer,
            loss_fn=MaskedGaussianLogLikelihood(),
            metrics={},
            callbacks=callbacks,
            verbose=cfg.training.verbose if is_main_process() else 0,
            strict=False,
            grad_clip=cfg.training.grad_clip,
        )
        logger.info(
            "Starting distributed training for up to %d epochs", cfg.training.max_epochs
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=cfg.training.max_epochs,
        )

        if is_main_process():
            # Unwrap DDP before saving so downstream code loads the raw module state.
            raw_model = getattr(trainer.cfg.model, "module", trainer.cfg.model)
            _save_training_artifacts(
                cfg=cfg, model=raw_model, resolved_params=resolved_params
            )
    finally:
        teardown()

    return cfg.artifacts.output_path().resolve()


def _build_callbacks(cfg: PanelPipelineConfig) -> list[Any]:
    callbacks: list[Any] = []
    if cfg.early_stopping.enabled:
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
    if cfg.model_checkpoint.enabled:
        checkpoint_path = cfg.artifacts.output_path() / "checkpoints" / "best_model.pt"
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor=cfg.model_checkpoint.monitor,
                mode=cfg.model_checkpoint.mode,
                min_delta=cfg.model_checkpoint.min_delta,
                warmup=cfg.model_checkpoint.warmup,
                save_optimizer=cfg.model_checkpoint.save_optimizer,
                overwrite=cfg.model_checkpoint.overwrite,
                verbose=cfg.model_checkpoint.verbose,
            )
        )
    return callbacks


def _save_training_artifacts(
    *, cfg: PanelPipelineConfig, model: torch.nn.Module, resolved_params: dict[str, Any]
) -> None:
    output_dir = cfg.artifacts.output_path().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_state_path = cfg.artifacts.model_state_path().resolve()
    torch.save(model.state_dict(), model_state_path)

    manifest = PanelTrainingManifest(
        preprocessing_path=str(cfg.artifacts.preprocessing_path().resolve()),
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
        target_mode=cfg.features.target_mode,
    )
    manifest.save(cfg.artifacts.manifest_path())
    logger.info("Saved model weights + manifest to %s", output_dir)


def _load_preprocessing_bundle(path: Path) -> PreprocessingBundle:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing artifact not found at {path}. "
            "Run scripts/preprocess.py before training."
        )
    artifact = BasePreprocessorArtifact.load(path)
    if not isinstance(artifact, PreprocessingBundle):
        raise TypeError(
            f"Preprocessing artifact at {path} must deserialize to PreprocessingBundle."
        )
    return artifact


def _load_vectorized_panel(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Vectorized panel not found at {path}. "
            "Run scripts/preprocess.py before training."
        )
    return torch.load(path, weights_only=False)


def _resolve_model_params(
    *,
    cfg: PanelPipelineConfig,
    preprocessing: PreprocessingBundle,
) -> dict[str, Any]:
    """Fill runtime-derived model params from the feature + preprocessing context."""

    params: dict[str, Any] = dict(cfg.model.params)

    params.setdefault(
        "continuous_dim",
        len(cfg.features.continuous_columns) if cfg.features.continuous_columns else None,
    )

    tokenizer = preprocessing.categorical_tokenizer
    dynamic_cols = cfg.features.dynamic_categorical_columns
    static_cols = cfg.features.static_categorical_columns
    if (dynamic_cols or static_cols) and tokenizer is None:
        raise ValueError(
            "Categorical streams require a fitted categorical tokenizer in preprocessing."
        )
    if dynamic_cols:
        params["dynamic_categorical_cardinalities"] = list(
            tokenizer.get_vocabulary_sizes(dynamic_cols)
        )
    if static_cols:
        params["static_categorical_cardinalities"] = list(
            tokenizer.get_vocabulary_sizes(static_cols)
        )
    return params
