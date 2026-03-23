"""
Training Workflow
-----------------
CSV-based orchestration for fitting preprocessing artifacts, building the
forecasting model from separate train and validation feature datasets, and
recording a frozen training manifest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from density_model.config import TrainConfig, TrainingArtifactManifest
from density_model.data import YahooVolatilityPanelBuilder
from density_model.datasets import VectorizedPanelDataset
from density_model.preprocessing import PreprocessingBundle

__all__ = ["train_from_config"]


def train_from_config(config: TrainConfig, *, resume_from: str | Path | None = None) -> Path:
    """
    Run the repository training workflow from an extracted feature CSV.

    Parameters
    ----------
    config : TrainConfig
        Typed CSV-based training configuration.
    resume_from : str or pathlib.Path or None, default=None
        Optional checkpoint path used to resume model and optimizer state.

    Returns
    -------
    pathlib.Path
        Artifact output directory used by the training run.
    """

    try:
        import torch
        from torch.optim import Adam
        from torch.utils.data import DataLoader
    except ImportError as error:  # pragma: no cover - exercised only when torch is unavailable.
        raise ImportError("torch must be installed to run the training workflow.") from error

    from density_model.models import build_panel_model
    from density_model.torch_datasets import VectorizedPanelTorchDataset
    from density_model.training import MaskedGaussianLogLikelihood, SupervisedTrainer
    from density_model.training.callbacks import (
        EarlyStopping,
        EarlyStoppingConfig,
        ModelCheckpoint,
        ModelCheckpointConfig,
    )

    torch.manual_seed(config.training.random_seed)
    panel_builder = YahooVolatilityPanelBuilder(
        id_column=config.dataset.id_column,
        date_column=config.dataset.date_column,
    )
    train_feature_panel = _load_feature_panel(
        csv_path=config.dataset.train_csv_file(),
        date_column=config.dataset.date_column,
        id_column=config.dataset.id_column,
        target_column=config.features.target_column,
    )
    validation_feature_panel = _load_feature_panel(
        csv_path=config.dataset.validation_csv_file(),
        date_column=config.dataset.date_column,
        id_column=config.dataset.id_column,
        target_column=config.features.target_column,
    )
    train_panel_raw = panel_builder.transform_feature_panel(
        panel=train_feature_panel,
        feature_config=config.features,
    )
    validation_panel_raw = panel_builder.transform_feature_panel(
        panel=validation_feature_panel,
        feature_config=config.features,
    )

    continuous_columns = config.features.streams.resolve_continuous_columns()
    dynamic_categorical_columns = config.features.streams.resolve_dynamic_categorical_columns()
    static_categorical_columns = config.features.streams.resolve_static_categorical_columns()

    preprocessing = PreprocessingBundle().fit(
        train_panel_raw,
        continuous_columns=continuous_columns,
        categorical_columns=dynamic_categorical_columns + static_categorical_columns,
    )
    train_vectorized = _build_vectorized_panel(
        panel=preprocessing.transform(train_panel_raw),
        config=config,
        continuous_columns=continuous_columns,
        dynamic_categorical_columns=dynamic_categorical_columns,
        static_categorical_columns=static_categorical_columns,
    )
    validation_vectorized = _build_vectorized_panel(
        panel=preprocessing.transform(validation_panel_raw),
        config=config,
        continuous_columns=continuous_columns,
        dynamic_categorical_columns=dynamic_categorical_columns,
        static_categorical_columns=static_categorical_columns,
    )
    if train_vectorized["y"] is None:
        raise ValueError("Training dataset did not produce any valid forecast windows.")
    if validation_vectorized["y"] is None:
        raise ValueError("Validation dataset did not produce any valid forecast windows.")

    train_dataset = VectorizedPanelTorchDataset(train_vectorized)
    validation_dataset = VectorizedPanelTorchDataset(validation_vectorized)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.training.batch_size, shuffle=False)

    model = build_panel_model(
        model_config=config.model,
        feature_config=config.features,
        preprocessing=preprocessing,
    )
    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    initial_epoch = 0
    if resume_from is not None:
        initial_epoch = _load_resume_checkpoint(
            resume_path=Path(resume_from),
            model=model,
            optimizer=optimizer,
            device=config.training.device,
        )
    callbacks = []
    if config.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                EarlyStoppingConfig(
                    monitor=config.early_stopping.monitor,
                    mode=config.early_stopping.mode,
                    patience=config.early_stopping.patience,
                    min_delta=config.early_stopping.min_delta,
                    warmup=config.early_stopping.warmup,
                    restore_best_weights=config.early_stopping.restore_best_weights,
                    verbose=config.early_stopping.verbose,
                )
            )
        )
    if config.model_checkpoint.enabled:
        callbacks.append(
            ModelCheckpoint(
                ModelCheckpointConfig(
                    filepath=str((config.artifacts.output_path() / "checkpoints").resolve()),
                    monitor=config.model_checkpoint.monitor,
                    mode=config.model_checkpoint.mode,
                    min_delta=config.model_checkpoint.min_delta,
                    warmup=config.model_checkpoint.warmup,
                    save_optimizer=config.model_checkpoint.save_optimizer,
                    overwrite=config.model_checkpoint.overwrite,
                    verbose=config.model_checkpoint.verbose,
                )
            )
        )
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=MaskedGaussianLogLikelihood(),
        metrics={},
        callbacks=callbacks,
        grad_clip=config.training.grad_clip,
        device=config.training.device,
        verbose=config.training.verbose,
        strict=True,
    )
    trainer.train(
        train_dataloader=train_dataloader,
        epochs=config.training.max_epochs,
        val_dataloader=validation_dataloader,
        initial_epoch=initial_epoch,
    )

    artifact_dir = config.artifacts.output_path()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = config.artifacts.preprocessing_path().resolve()
    model_state_path = config.artifacts.model_state_path().resolve()
    preprocessing.save(preprocessing_path)
    torch.save(model.state_dict(), model_state_path)
    manifest = TrainingArtifactManifest(
        feature_config=config.features,
        model_config_runtime=config.model,
        calendar="XNYS",
        preprocessing_path=str(preprocessing_path),
        model_state_path=str(model_state_path),
    )
    manifest.save(config.artifacts.manifest_path())
    return artifact_dir


def _load_resume_checkpoint(
    *,
    resume_path: Path,
    model: object,
    optimizer: object,
    device: str,
) -> int:
    """
    Load model and optimizer state from a saved checkpoint.

    Parameters
    ----------
    resume_path : pathlib.Path
        Checkpoint path to restore.
    model : object
        Model instance to update in place.
    optimizer : object
        Optimizer instance to update in place.
    device : str
        Torch device used to read the checkpoint file.

    Returns
    -------
    int
        Next zero-based epoch index from which training should continue.

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    ValueError
        If the checkpoint payload is missing required state.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only when torch is unavailable.
        raise ImportError("torch must be installed to resume a training run.") from error

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint was not found: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint must deserialize to a mapping.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint is missing 'model_state_dict'.")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(
            "Resume checkpoint is missing 'optimizer_state_dict'. Enable model_checkpoint.save_optimizer "
            "to support exact training resume."
        )
    if "epoch" not in checkpoint:
        raise ValueError("Resume checkpoint is missing 'epoch'.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer=optimizer, device=device)
    resume_epoch = int(checkpoint["epoch"]) + 1
    if resume_epoch < 0:
        raise ValueError("Resume checkpoint produced an invalid negative epoch index.")
    return resume_epoch


def _move_optimizer_state_to_device(*, optimizer: object, device: str) -> None:
    """
    Move optimizer state tensors onto the requested torch device.

    Parameters
    ----------
    optimizer : object
        Optimizer instance whose internal state is updated in place.
    device : str
        Torch device identifier.

    Returns
    -------
    None
        This method mutates optimizer state tensors in place.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised only when torch is unavailable.
        raise ImportError("torch must be installed to move optimizer state tensors.") from error

    resolved_device = torch.device(device)
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(resolved_device)


def _load_feature_panel(
    *,
    csv_path: Path,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Load and validate an extracted feature CSV.

    Parameters
    ----------
    csv_path : pathlib.Path
        Extracted feature CSV path.
    date_column : str
        Date column name.
    id_column : str
        Entity identifier column name.
    target_column : str
        Supervised target column name.

    Returns
    -------
    pandas.DataFrame
        Canonical feature panel sorted by identifier and date.
    """

    frame = pd.read_csv(csv_path)
    required_columns = {date_column, id_column, "return", "ticker", target_column}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Training dataset is missing required columns: {missing_text}")
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)


def _build_vectorized_panel(
    *,
    panel: pd.DataFrame,
    config: TrainConfig,
    continuous_columns: tuple[str, ...],
    dynamic_categorical_columns: tuple[str, ...],
    static_categorical_columns: tuple[str, ...],
) -> dict[str, object]:
    """
    Build vectorized panel tensors from a transformed long-format panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Transformed long-format panel.
    config : TrainConfig
        Typed training configuration.
    continuous_columns : tuple of str
        Continuous feature columns.
    dynamic_categorical_columns : tuple of str
        Dynamic categorical feature columns.
    static_categorical_columns : tuple of str
        Static categorical feature columns.

    Returns
    -------
    dict of str to object
        Vectorized panel output.
    """

    return VectorizedPanelDataset(
        data=panel,
        date_column=config.dataset.date_column,
        id_column=config.dataset.id_column,
        targets=config.features.target_column,
        in_steps=config.features.sequence_length,
        horizon=config.features.forecast_horizon,
        out_steps=1,
        dynamic_features=continuous_columns,
        dynamic_categorical_features=dynamic_categorical_columns or None,
        static_categorical_features=static_categorical_columns or None,
    ).generate_sequences()
