"""
Training Tests
--------------
Unit tests for trainer validation and the base training loop contract.
"""

from __future__ import annotations

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from pydantic import ValidationError
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from density_model.models import CausalCrossAssetTransformer, CausalCrossAssetTransformerConfig
from density_model.torch_datasets import VectorizedPanelTorchDataset
from density_model.training import BaseTrainer, Callback, SupervisedTrainer
from density_model.training import MaskedGaussianLogLikelihood
from density_model.workflows.training import _load_resume_checkpoint

__all__ = []


class MeanAbsoluteError(nn.Module):
    """
    Metric module computing mean absolute error.

    Parameters
    ----------
    None
        This metric has no configurable parameters.
    """

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute mean absolute error.

        Parameters
        ----------
        y_true : torch.Tensor
            Ground-truth target tensor.
        y_pred : torch.Tensor
            Predicted target tensor.

        Returns
        -------
        torch.Tensor
            Scalar mean absolute error.
        """

        return torch.mean(torch.abs(y_true - y_pred))


def mean_squared_metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error as a plain callable metric.

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted target tensor.
    target : torch.Tensor
        Ground-truth target tensor.

    Returns
    -------
    torch.Tensor
        Scalar mean squared error.
    """

    return torch.mean((prediction - target) ** 2)


class RecorderCallback(Callback):
    """
    Test callback that records trainer lifecycle events.

    Parameters
    ----------
    None
        State is recorded internally during training.
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_fit_start(self) -> None:
        """Record fit start."""

        self.events.append("fit_start")

    def on_epoch_end(self, epoch: int, logs: dict[str, float]) -> None:
        """
        Record epoch end.

        Parameters
        ----------
        epoch : int
            Zero-based epoch index.
        logs : dict of str to float
            Aggregated trainer logs.
        """

        self.events.append(f"epoch_end_{epoch}")

    def on_fit_end(self) -> None:
        """Record fit end."""

        self.events.append("fit_end")


class BrokenTrainer(BaseTrainer):
    """
    Trainer used to test strict metric behavior in the base loop.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments forwarded to ``BaseTrainer``.
    """

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Return only loss metrics to trigger strict metric validation.

        Parameters
        ----------
        batch : tuple of torch.Tensor and torch.Tensor
            Training batch.

        Returns
        -------
        dict of str to torch.Tensor
            Loss-only metric mapping.
        """

        features, target = batch
        prediction = self.model(features)
        loss = self.loss_fn(prediction, target)
        return {"loss": loss}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Return only loss metrics to trigger strict metric validation.

        Parameters
        ----------
        batch : tuple of torch.Tensor and torch.Tensor
            Validation batch.

        Returns
        -------
        dict of str to torch.Tensor
            Loss-only metric mapping.
        """

        features, target = batch
        prediction = self.model(features)
        loss = self.loss_fn(prediction, target)
        return {"loss": loss}


def build_components() -> tuple[nn.Module, SGD, nn.Module]:
    """
    Build minimal model, optimizer, and loss objects for trainer tests.

    Returns
    -------
    tuple
        Model, optimizer, and loss instances.
    """

    model = nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn


def build_dataloader() -> DataLoader:
    """
    Build a deterministic single-feature regression dataloader.

    Returns
    -------
    torch.utils.data.DataLoader
        Small dataloader used by trainer tests.
    """

    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def build_partial_batch_dataloader() -> DataLoader:
    """
    Build a dataloader whose last batch is smaller than the nominal batch size.

    Returns
    -------
    torch.utils.data.DataLoader
        Dataloader used to test sample-weighted aggregation.
    """

    features = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_supervised_trainer_runs_base_training_loop() -> None:
    """
    Run the base training loop through the concrete supervised trainer.

    Returns
    -------
    None
        This test asserts that logs and callbacks are produced.
    """

    model, optimizer, loss_fn = build_components()
    callback = RecorderCallback()
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics={"mae": MeanAbsoluteError(), "mse": mean_squared_metric},
        callbacks=[callback],
        device="cpu",
        verbose=0,
        strict=True,
    )

    dataloader = build_dataloader()
    trainer.train(train_dataloader=dataloader, epochs=2, val_dataloader=dataloader)

    history = trainer.logger.history
    assert len(history) == 2
    assert "val_loss" in history.columns
    assert "train_mae" in trainer.logger.last_log()
    assert "train_mse" in trainer.logger.last_log()
    assert callback.events == ["fit_start", "epoch_end_0", "epoch_end_1", "fit_end"]


def test_load_resume_checkpoint_restores_model_optimizer_and_next_epoch(tmp_path: Path) -> None:
    """
    Restore model state, optimizer state, and next epoch from a checkpoint.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test asserts exact resume semantics.
    """

    model, optimizer, _ = build_components()
    checkpoint_path = tmp_path / "resume.pt"
    checkpoint = {
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

    resumed_model, resumed_optimizer, _ = build_components()
    next_epoch = _load_resume_checkpoint(
        resume_path=checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        device="cpu",
    )

    assert next_epoch == 4
    for original_param, resumed_param in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(original_param, resumed_param)


def test_trainer_validation_rejects_invalid_verbose_value() -> None:
    """
    Reject invalid trainer verbosity values during pydantic validation.

    Returns
    -------
    None
        This test asserts fail-fast configuration validation.
    """

    model, optimizer, loss_fn = build_components()
    with pytest.raises(ValidationError):
        SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            verbose=4,
        )


def test_supervised_trainer_runs_end_to_end_panel_model_step() -> None:
    """
    Run one end-to-end epoch through the panel dataset, model, and Gaussian loss.

    Returns
    -------
    None
        This test asserts the panel-model training path is coherent.
    """

    panel_data = {
        "X_continuous": torch.tensor(
            [
                [[[0.1], [0.2]], [[0.3], [0.4]]],
                [[[0.5], [0.6]], [[0.7], [0.8]]],
            ],
            dtype=torch.float32,
        ).numpy(),
        "X_continuous_mask": torch.ones(2, 2, 2, 1, dtype=torch.bool).numpy(),
        "X_cat_continuous": None,
        "X_cat_continuous_mask": None,
        "X_cat_static": None,
        "X_cat_static_mask": None,
        "y": torch.tensor(
            [
                [[[0.15]], [[0.35]]],
                [[[0.55]], [[0.75]]],
            ],
            dtype=torch.float32,
        ).numpy(),
        "y_mask": torch.tensor(
            [
                [[[True]], [[True]]],
                [[[True]], [[False]]],
            ],
            dtype=torch.bool,
        ).numpy(),
        "id_index": ["SPY", "QQQ"],
        "forecast_start_dates": np.array(["2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
        "forecast_end_dates": np.array(["2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
    }
    dataset = VectorizedPanelTorchDataset(panel_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CausalCrossAssetTransformer(
        CausalCrossAssetTransformerConfig(
            d_model=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            fusion_mode="sum",
            out_steps=1,
            target_dim=1,
        ),
        continuous_dim=1,
    )
    optimizer = SGD(model.parameters(), lr=0.01)
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=MaskedGaussianLogLikelihood(),
        metrics={},
        device="cpu",
        verbose=0,
        strict=True,
    )

    trainer.train(train_dataloader=dataloader, epochs=1, val_dataloader=dataloader)

    history = trainer.logger.history
    assert len(history) == 1
    assert "train_loss" in history.columns
    assert "val_loss" in history.columns


def test_trainer_validation_rejects_invalid_device_value() -> None:
    """
    Reject invalid torch device identifiers during pydantic validation.

    Returns
    -------
    None
        This test asserts fail-fast device validation.
    """

    model, optimizer, loss_fn = build_components()
    with pytest.raises(ValidationError):
        SupervisedTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device="definitely_not_a_device",
        )


def test_base_trainer_strict_metrics_require_y_true_and_y_pred() -> None:
    """
    Raise when strict metrics are configured without ``y_true`` and ``y_pred`` outputs.

    Returns
    -------
    None
        This test asserts the base loop metric contract.
    """

    model, optimizer, loss_fn = build_components()
    trainer = BrokenTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics={"mae": MeanAbsoluteError()},
        device="cpu",
        verbose=0,
        strict=True,
    )

    dataloader = build_dataloader()
    try:
        trainer.train(train_dataloader=dataloader, epochs=1)
    except KeyError as error:
        assert "y_true" in str(error)
    else:
        raise AssertionError("Expected strict metric validation to raise KeyError.")


def test_epoch_metrics_use_actual_partial_batch_size() -> None:
    """
    Weight epoch metrics by the true batch size of each batch.

    Returns
    -------
    None
        This test asserts correct aggregation for a smaller final batch.
    """

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    optimizer = SGD(model.parameters(), lr=0.0)
    loss_fn = nn.MSELoss()

    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        verbose=0,
        strict=True,
    )
    dataloader = build_partial_batch_dataloader()
    trainer.train(train_dataloader=dataloader, epochs=1)

    logged_loss = trainer.logger.history.loc[0, "loss"]
    expected_loss = float((0.5 + 4.0) / 3.0)
    assert logged_loss == pytest.approx(expected_loss)


def test_move_to_device_preserves_nested_batch_structure() -> None:
    """
    Move nested tensor batches onto the configured device.

    Returns
    -------
    None
        This test asserts recursive device placement behavior.
    """

    model, optimizer, loss_fn = build_components()
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        verbose=0,
    )

    batch = (
        {"x": torch.tensor([[1.0]], dtype=torch.float32)},
        [torch.tensor([[2.0]], dtype=torch.float32)],
    )
    moved = trainer.move_to_device(batch)

    assert isinstance(moved, tuple)
    assert moved[0]["x"].device.type == "cpu"
    assert moved[1][0].device.type == "cpu"
