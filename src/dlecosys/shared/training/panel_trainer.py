"""
Panel Trainer
-------------
Concrete panel-forecasting trainer built on top of :class:`BaseTrainer`.
Consumes DataLoader batches of the form ``(features_dict, y, y_mask)`` emitted
by ``BasePanelTorchDataset`` subclasses, wraps each batch into a validated
``PanelBatch`` before the model forward pass, and passes the target mask to
the loss function.

Classes
-------
PanelTrainer
    Subclass of ``BaseTrainer`` with panel-aware ``_train_epoch`` /
    ``_validate_epoch`` (batch-size inference and device movement handle the
    three-tuple dict-leading layout) and ``train_step`` / ``validation_step``
    that build a ``PanelBatch`` and apply masked losses.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch.nn.utils import clip_grad_norm_

from dlecosys.models.panel_base import BasePanelModel
from dlecosys.shared.data.panel.schema import PanelBatch
from dlecosys.shared.training.base import BaseTrainer
from dlecosys.shared.training.utils import ProgressBar

__all__ = ["PanelTrainer"]


def _panel_batch_size(batch: Any) -> int:
    """Return the leading batch axis of a ``(features, y, y_mask)`` panel batch."""

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        y = batch[1]
        if hasattr(y, "shape") and y.ndim >= 1:
            return int(y.shape[0])
    raise ValueError(
        "PanelTrainer expects batches of the form (features_dict, y, y_mask)."
    )


class PanelTrainer(BaseTrainer):
    """
    Panel-aware supervised trainer.

    Parameters
    ----------
    grad_clip : float or None, default=1.0
        Maximum gradient norm for clipping; ``None`` disables clipping.
    **kwargs : dict
        Forwarded to ``BaseTrainer`` (``model``, ``optimizer``, ``loss_fn``,
        ``metrics``, ``callbacks``, ``verbose``, ``strict``, ``device``).

    Notes
    -----
    - The loss function is called as ``loss_fn(y_pred, y_true, y_mask)``. Use
      :class:`MaskedGaussianLogLikelihood` or any other mask-aware loss.
    - Model must be a ``BasePanelModel`` subclass accepting a ``PanelBatch``.
    """

    def __init__(self, grad_clip: float | None = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.grad_clip = grad_clip

    def _train_epoch(self, epoch: int, train_dataloader, progress_bar: ProgressBar) -> None:
        """Run one training epoch with panel-aware batch-size inference."""

        declared_bs = getattr(train_dataloader, "batch_size", None)
        try:
            steps_per_epoch = len(train_dataloader)
        except Exception:
            steps_per_epoch = 0

        self.cfg.model.train()
        samples_seen = 0
        totals: Dict[str, float] = {}

        for step, batch in enumerate(train_dataloader, start=1):
            bs = _panel_batch_size(batch)
            if declared_bs is not None and bs < declared_bs:
                continue

            self.callbacks.on_train_step_start(step, batch)

            step_metrics, info = self._run_training_step(batch)
            step_metrics = {
                k: (float(v.item()) if hasattr(v, "item") else float(v))
                for k, v in step_metrics.items()
            }

            samples_seen += bs
            step_metrics.update(self._compute_metrics(info, prefix="train_"))

            for k, v in step_metrics.items():
                totals[k] = totals.get(k, 0.0) + v * bs
            self.logger.update_state(**{k: totals[k] / samples_seen for k in totals})

            self.callbacks.on_train_step_end(
                step=step,
                batch=batch,
                outputs={"metrics": step_metrics, "info": info},
                logs=self.logger.last_log(),
            )

            if self.verbose == 2 and steps_per_epoch:
                progress_bar(epoch, step, self.logger.last_log())

    def _validate_epoch(self, epoch: int, val_dataloader, progress_bar: ProgressBar) -> None:
        """Run one validation epoch with panel-aware batch-size inference.

        Validation uses a dedicated :class:`ProgressBar` instance so its line
        renders independently from the training bar and inherits its own ETA
        clock instead of being reset in place.
        """

        try:
            val_steps = len(val_dataloader)
        except Exception:
            val_steps = 0

        # Finalize the training bar on its own line before switching to validation.
        if self.verbose == 2:
            progress_bar.end_epoch(epoch, self.logger.last_log())

        total_epochs = getattr(progress_bar, "total_epochs", 0) or 0
        val_progress = ProgressBar(
            name="Validation",
            total_epochs=total_epochs,
            steps_per_epoch=val_steps,
            length=progress_bar.length,
            fill=progress_bar.fill,
            eta_smoothing=progress_bar.eta_smoothing,
        )

        self.cfg.model.eval()
        samples_seen = 0
        totals: Dict[str, float] = {}

        with torch.no_grad():
            for vstep, batch in enumerate(val_dataloader, start=1):
                self.callbacks.on_validation_step_start(vstep, batch)

                step_metrics, info = self._run_validation_step(batch)
                step_metrics = {
                    f"val_{k}": (float(v.item()) if hasattr(v, "item") else float(v))
                    for k, v in step_metrics.items()
                }

                bs = _panel_batch_size(batch)
                samples_seen += bs
                step_metrics.update(self._compute_metrics(info, prefix="val_"))

                for k, v in step_metrics.items():
                    totals[k] = totals.get(k, 0.0) + v * bs
                self.logger.update_state(**{k: totals[k] / samples_seen for k in totals})

                self.callbacks.on_validation_step_end(
                    vstep=vstep,
                    batch=batch,
                    outputs={"metrics": step_metrics, "info": info},
                    logs=self.logger.last_log(),
                )

                if self.verbose == 2 and val_steps:
                    val_progress(epoch, vstep, self.logger.last_log())

        if self.verbose > 0:
            val_progress.end_epoch(epoch, self.logger.last_log())

    def _move_to_device(self, batch: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
        """Move a ``(features_dict, y, y_mask)`` batch to the configured device."""

        if self.cfg.device is None:
            return batch
        features, y, y_mask = batch
        features_on_device = {
            key: (value.to(self.cfg.device) if value is not None else None)
            for key, value in features.items()
        }
        y_on_device = y.to(self.cfg.device)
        y_mask_on_device = y_mask.to(self.cfg.device)
        return features_on_device, y_on_device, y_mask_on_device

    def train_step(
        self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform a single training step on a panel batch."""

        features, y, y_mask = batch
        panel_batch = self._build_panel_batch(features=features, y=y, y_mask=y_mask)

        self.optimizer.zero_grad()
        y_hat = self.cfg.model(panel_batch)
        loss = self.loss_fn(y_hat, y, y_mask)
        loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.cfg.model.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        info = {"y_true": y, "y_pred": y_hat, "y_mask": y_mask}
        return {"loss": loss}, info

    def validation_step(
        self, batch: Tuple[Any, Any, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform a single validation step on a panel batch."""

        features, y, y_mask = batch
        panel_batch = self._build_panel_batch(features=features, y=y, y_mask=y_mask)

        with torch.inference_mode():
            y_hat = self.cfg.model(panel_batch)
            loss = self.loss_fn(y_hat, y, y_mask)

        info = {"y_true": y, "y_pred": y_hat, "y_mask": y_mask}
        return {"loss": loss}, info

    def _build_panel_batch(
        self,
        *,
        features: Dict[str, Any],
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> PanelBatch:
        """Wrap a trainer-facing feature dict + target into a validated ``PanelBatch``."""

        if not isinstance(self.cfg.model, BasePanelModel):
            raise TypeError(
                "PanelTrainer requires the model to subclass BasePanelModel."
            )
        if not isinstance(features, dict):
            raise TypeError(
                "PanelTrainer expects the feature payload to be a dict keyed by stream name."
            )
        return PanelBatch.from_training_batch(features=features, y=y, y_mask=y_mask)
