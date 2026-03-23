"""
Supervised Trainer
------------------
Concrete supervised trainer built on top of ``BaseTrainer`` with explicit
``train_step`` and ``validation_step`` implementations.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch.nn.utils import clip_grad_norm_

from density_model.models.base import BasePanelModel
from density_model.models.schema import PanelBatch
from density_model.training.base import BaseTrainer

__all__ = ["SupervisedTrainer", "SupervisedTrainerConfig"]


class SupervisedTrainerConfig(BaseModel):
    """
    Pydantic configuration for supervised trainer behavior.

    Parameters
    ----------
    grad_clip : float or None, default=1.0
        Maximum gradient norm used for clipping. ``None`` disables clipping.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    grad_clip: float | None = 1.0

    @field_validator("grad_clip")
    @classmethod
    def validate_grad_clip(cls, value: float | None) -> float | None:
        """
        Validate the gradient clipping threshold.

        Parameters
        ----------
        value : float or None
            Requested gradient clipping threshold.

        Returns
        -------
        float or None
            Validated clipping threshold.

        Raises
        ------
        ValueError
            If the clipping threshold is not strictly positive.
        """

        if value is not None and value <= 0.0:
            raise ValueError("grad_clip must be strictly positive when provided.")
        return value


class SupervisedTrainer(BaseTrainer):
    """
    Concrete supervised trainer implementing batch-level train and validation steps.

    Parameters
    ----------
    grad_clip : float or None, default=1.0
        Maximum gradient norm used for clipping. ``None`` disables clipping.
    **kwargs : dict
        Keyword arguments forwarded to ``BaseTrainer``.
    """

    def __init__(self, grad_clip: float | None = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.step_config = SupervisedTrainerConfig(grad_clip=grad_clip)

    def train_step(self, batch: tuple[Any, ...]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Execute one supervised training step.

        Parameters
        ----------
        batch : tuple of Any and Any
            Tuple ``(X, y)`` where ``X`` is the model input and ``y`` is the target.

        Returns
        -------
        tuple of dict and dict
            Metrics and auxiliary information dictionaries.
        """

        features, target, target_mask = self._unpack_batch(self.move_to_device(batch))
        self.optimizer.zero_grad()
        model_input = self._prepare_model_input(features=features, target=target, target_mask=target_mask)
        prediction = self.model(*model_input) if isinstance(model_input, (tuple, list)) else self.model(model_input)
        loss = self._compute_loss(prediction=prediction, target=target, target_mask=target_mask)
        loss.backward()
        if self.step_config.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), max_norm=self.step_config.grad_clip)
        self.optimizer.step()
        info = {"y_true": target, "y_pred": prediction}
        if target_mask is not None:
            info["y_mask"] = target_mask
        return {"loss": loss}, info

    def validation_step(
        self,
        batch: tuple[Any, ...],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Execute one supervised validation step.

        Parameters
        ----------
        batch : tuple of Any and Any
            Tuple ``(X, y)`` where ``X`` is the model input and ``y`` is the target.

        Returns
        -------
        tuple of dict and dict
            Metrics and auxiliary information dictionaries.
        """

        features, target, target_mask = self._unpack_batch(self.move_to_device(batch))
        with torch.inference_mode():
            model_input = self._prepare_model_input(features=features, target=target, target_mask=target_mask)
            prediction = self.model(*model_input) if isinstance(model_input, (tuple, list)) else self.model(model_input)
            loss = self._compute_loss(prediction=prediction, target=target, target_mask=target_mask)
        info = {"y_true": target, "y_pred": prediction}
        if target_mask is not None:
            info["y_mask"] = target_mask
        return {"loss": loss}, info

    def _unpack_batch(self, batch: tuple[Any, ...]) -> tuple[Any, Any, Any | None]:
        """
        Unpack a supervised batch into features, targets, and an optional target mask.

        Parameters
        ----------
        batch : tuple
            Expected ``(features, target)`` or ``(features, target, target_mask)`` batch structure.

        Returns
        -------
        tuple
            Feature payload, supervision target, and optional target mask.

        Raises
        ------
        ValueError
            If the batch does not contain two or three elements.
        """

        if not isinstance(batch, (tuple, list)) or len(batch) not in {2, 3}:
            raise ValueError(
                "SupervisedTrainer expects each batch to be (features, target) or "
                "(features, target, target_mask)."
            )
        if len(batch) == 2:
            features, target = batch
            return features, target, None
        features, target, target_mask = batch
        return features, target, target_mask

    def _compute_loss(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute training loss with optional target masking.

        Parameters
        ----------
        prediction : torch.Tensor
            Predicted target tensor.
        target : torch.Tensor
            Ground-truth target tensor.
        target_mask : torch.Tensor or None
            Optional target-availability mask.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """

        if target_mask is None:
            return self.loss_fn(prediction, target)
        return self.loss_fn(prediction, target, target_mask)

    def _prepare_model_input(
        self,
        *,
        features: Any,
        target: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> Any:
        """
        Adapt trainer features to the model's expected forward input.

        Parameters
        ----------
        features : Any
            Trainer-facing feature payload.
        target : torch.Tensor
            Ground-truth target tensor.
        target_mask : torch.Tensor or None
            Optional target mask tensor.

        Returns
        -------
        Any
            Model-ready forward input payload.
        """

        if isinstance(self.model, BasePanelModel):
            if not isinstance(features, dict):
                raise TypeError("BasePanelModel instances require feature dictionaries compatible with PanelBatch.")
            resolved_target_mask = (
                target_mask if target_mask is not None else torch.ones_like(target, dtype=torch.bool)
            )
            return PanelBatch.from_training_batch(
                features=features,
                y=target,
                y_mask=resolved_target_mask,
            )
        return features
