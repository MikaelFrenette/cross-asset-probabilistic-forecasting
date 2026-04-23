"""
Trainer Class
==============

This module defines the :class:`Trainer`, a concrete implementation of
:class:`BaseTrainer`, providing a standard supervised learning loop with
gradient clipping, optional AMP, and separate training/validation steps.

Classes
-------
Trainer
    A lightweight subclass of :class:`BaseTrainer` that performs forward
    propagation, loss computation, backward propagation, and optimization with
    optional gradient clipping and automatic mixed precision (AMP). It supports
    multiple input tensors and returns structured metric and info dictionaries
    for logging and analysis.

Notes
-----
- The class assumes the dataset returns batches of the form ``(X, y)``, where
  ``X`` can be a tensor or tuple of tensors matching the model’s input
  signature.
- Gradient clipping is performed using :func:`torch.nn.utils.clip_grad_norm_`.
- When AMP is enabled (``amp=True`` in config), the forward pass runs under
  ``torch.autocast`` and a ``GradScaler`` manages loss scaling on CUDA. The
  scaler is automatically disabled when the device is not CUDA.
- Validation steps are executed under ``torch.inference_mode()`` to disable
  autograd and reduce memory usage.
- The trainer inherits common functionality (e.g., logging, device handling)
  from :class:`BaseTrainer`.

Author  : Mikael Frenette
Contact : mik.frenette@gmail.com
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_

from ..training.base import BaseTrainer

__all__ = ["Trainer"]


class Trainer(BaseTrainer):
    """
    A lightweight training loop built on top of ``BaseTrainer`` that performs a
    single optimization step per batch with optional gradient clipping and AMP.

    Parameters
    ----------
    grad_clip : float, default=1.0
        Maximum norm for gradient clipping (L2 norm). If ``None``, no clipping is
        applied.
    **kwargs : dict
        Additional keyword arguments forwarded to ``BaseTrainer`` (e.g., ``model``,
        ``optimizer``, ``loss_fn``, ``device``, ``verbose``, ``amp``).

    Attributes
    ----------
    grad_clip : float or None
        Maximum gradient norm for clipping; ``None`` disables clipping.
    model : torch.nn.Module
        PyTorch model to be trained (inherited from ``BaseTrainer``).
    optimizer : torch.optim.Optimizer
        Optimizer instance used for parameter updates (inherited).
    loss_fn : callable
        Loss function used to compute training/validation loss (inherited).

    Notes
    -----
    - Each call to :meth:`train_step` assumes the batch is a tuple ``(X, y)``,
      where ``X`` is either a tuple/list of model inputs or a single input tensor,
      and ``y`` is the supervision target.
    - The model is invoked with ``self.model(*X)`` to support multi-input models.
      Ensure your dataset/dataloader returns inputs consistent with the model
      signature.
    - :meth:`validation_step` executes under ``torch.inference_mode()`` and does
      not perform backpropagation.
    - Both methods return a pair ``(metrics, info)`` where:
        * ``metrics`` is a dict containing scalar tensors (e.g., ``{"loss": ...}``).
        * ``info`` is a dict carrying raw tensors useful for logging or inspection
          (e.g., ``{"y_true": y, "y_pred": y_hat}``).
    """

    def __init__(self, grad_clip: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.grad_clip = grad_clip
        _scaler_enabled = self.cfg.amp and getattr(self.cfg.device, "type", None) == "cuda"
        self._scaler = GradScaler("cuda", enabled=_scaler_enabled)
        self._accum_count: int = 0

    def _on_train_epoch_start(self, epoch: int) -> None:
        self._accum_count = 0

    def _move_to_device(self, batch: Tuple[Any, Any]) -> Tuple[Any, Any]:
        """Move an (X, y) batch to the configured device."""
        X, y = batch
        device = self.cfg.device
        if isinstance(X, (tuple, list)):
            X = tuple(t.to(device) if hasattr(t, "to") else t for t in X)
        elif hasattr(X, "to"):
            X = X.to(device)
        if hasattr(y, "to"):
            y = y.to(device)
        return X, y

    def train_step(self, batch: Tuple[Any, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform a single training step: forward, loss, backward, optional clip, step.

        Parameters
        ----------
        batch : tuple
            A tuple ``(X, y)`` where:
            - ``X`` : tuple | list | torch.Tensor
              Model inputs. Unpacked as ``*X`` for multi-input models.
            - ``y`` : torch.Tensor
              Target tensor.

        Returns
        -------
        metrics : dict
            Scalar metrics for logging (e.g., ``{"loss": loss}``).
        info : dict
            Raw tensors for callbacks and metric computation
            (e.g., ``{"y_true": y, "y_pred": y_hat}``).

        Notes
        -----
        When ``amp=True``, the forward pass runs under ``torch.autocast`` and
        loss scaling is handled by the internal ``GradScaler``. Gradient clipping
        is applied after unscaling so the clipped norm is in the original scale.

        With ``grad_accumulation_steps=N``, gradients accumulate over N batches
        and the optimizer steps only at the Nth batch. The loss is divided by N
        before the backward pass so the effective gradient magnitude is unchanged.
        The returned loss is unscaled (as if accumulation were not in use) so that
        the metric logger always shows the per-sample loss.
        """
        X, y = batch
        _device_type = self.cfg.device.type if self.cfg.device is not None else "cpu"
        accum_steps = self.cfg.grad_accumulation_steps

        self._accum_count += 1
        is_first_in_window = (self._accum_count - 1) % accum_steps == 0
        is_accum_boundary = self._accum_count % accum_steps == 0

        if is_first_in_window:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=_device_type, enabled=self.cfg.amp):
            y_hat = self.model(*X) if isinstance(X, (tuple, list)) else self.model(X)
            loss = self.loss_fn(y_hat, y) / accum_steps

        self._scaler.scale(loss).backward()

        if is_accum_boundary:
            if self.grad_clip is not None:
                self._scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()

        return {"loss": loss * accum_steps}, {"y_true": y, "y_pred": y_hat}

    def validation_step(self, batch: Tuple[Any, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform a single validation step without gradient updates.

        Parameters
        ----------
        batch : tuple
            A tuple ``(X, y)`` where:
            - ``X`` : tuple | list | torch.Tensor
              Model inputs to be forwarded through the network.
            - ``y`` : torch.Tensor
              Target tensor.

        Returns
        -------
        metrics : dict
            Scalar validation metrics (e.g., ``{"loss": loss}``).
        info : dict
            Raw tensors for monitoring (e.g., ``{"y_true": y, "y_pred": y_hat}``).

        Notes
        -----
        Runs under both ``torch.inference_mode()`` and ``torch.autocast`` (when
        ``amp=True``) for consistent dtype behaviour with the training pass.
        """
        X, y = batch
        _device_type = self.cfg.device.type if self.cfg.device is not None else "cpu"

        with torch.inference_mode():
            with torch.autocast(device_type=_device_type, enabled=self.cfg.amp):
                y_hat = self.model(*X) if isinstance(X, (tuple, list)) else self.model(X)
                loss = self.loss_fn(y_hat, y)

        return {"loss": loss}, {"y_true": y, "y_pred": y_hat}
