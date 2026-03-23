"""
Masked Loss Tests
-----------------
Unit tests for masked regression losses and masked supervised training batches.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.optim import SGD

from density_model.training import GaussianLogLikelihood, MaskedGaussianLogLikelihood, MaskedMSELoss, SupervisedTrainer

__all__ = []


def test_masked_mse_loss_ignores_invalid_positions() -> None:
    """
    Compute mean squared error only over valid target positions.

    Returns
    -------
    None
        This test asserts masked regression behavior.
    """

    prediction = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[1.0, 10.0]])
    mask = torch.tensor([[True, False]])

    loss = MaskedMSELoss()(prediction, target, mask)
    assert float(loss.item()) == 0.0


def test_gaussian_log_likelihood_splits_mu_and_sigma() -> None:
    """
    Compute Gaussian log-likelihood from concatenated ``mu`` and ``sigma`` predictions.

    Returns
    -------
    None
        This test asserts the probabilistic loss contract.
    """

    y_true = torch.tensor([[[[1.0]]]])
    mu = torch.tensor([[[[1.5]]]])
    sigma = torch.tensor([[[[2.0]]]])
    y_pred = torch.cat([mu, sigma], dim=-1)

    loss = GaussianLogLikelihood()(y_pred, y_true)
    expected = (((1.0 - 1.5) ** 2) / (2.0**2)) + torch.log(torch.tensor(2.0**2))
    assert torch.allclose(loss, expected)


def test_masked_gaussian_log_likelihood_ignores_invalid_positions() -> None:
    """
    Compute masked Gaussian log-likelihood over valid target positions only.

    Returns
    -------
    None
        This test asserts masked probabilistic supervision.
    """

    y_true = torch.tensor([[[[1.0]], [[2.0]]]])
    mu = torch.tensor([[[[1.0]], [[0.0]]]])
    sigma = torch.tensor([[[[1.0]], [[1.0]]]])
    y_pred = torch.cat([mu, sigma], dim=-1)
    y_mask = torch.tensor([[[[True]], [[False]]]])

    loss = MaskedGaussianLogLikelihood()(y_pred, y_true, y_mask)
    assert float(loss.item()) == 0.0


def test_supervised_trainer_accepts_optional_target_mask() -> None:
    """
    Accept three-item batches containing a target-availability mask.

    Returns
    -------
    None
        This test asserts masked-loss integration in the trainer.
    """

    model = nn.Linear(1, 1, bias=False)
    optimizer = SGD(model.parameters(), lr=0.0)
    loss_fn = MaskedMSELoss()
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device="cpu",
        verbose=0,
    )

    features = torch.tensor([[1.0], [2.0]])
    target = torch.tensor([[0.0], [0.0]])
    target_mask = torch.tensor([[True], [False]])

    metrics, info = trainer.train_step((features, target, target_mask))
    assert "loss" in metrics
    assert "y_mask" in info
