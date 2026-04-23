"""
Panel Distributed Trainer
-------------------------
DistributedDataParallel wrapper around :class:`PanelTrainer`. Mirrors
dlecosys's tabular ``DistributedTrainer`` contract but extends the panel
trainer so the panel-aware ``train_step`` / ``validation_step`` /
``_move_to_device`` / epoch-loop methods are preserved.

Classes
-------
PanelDistributedTrainer
    Subclass of ``PanelTrainer`` that wraps ``self.model`` in
    ``DistributedDataParallel`` after parent initialisation and calls
    ``sampler.set_epoch(epoch)`` at the start of each training epoch so each
    rank sees a different data shard per epoch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from dlecosys.shared.data.panel.schema import PanelBatch
from dlecosys.shared.training.panel_trainer import PanelTrainer
from dlecosys.shared.training.process_group import get_local_rank

__all__ = ["PanelDistributedTrainer"]


class PanelDistributedTrainer(PanelTrainer):
    """
    Panel-aware supervised trainer with ``DistributedDataParallel`` support.

    Parameters
    ----------
    train_sampler : DistributedSampler
        Sampler attached to the training DataLoader. Must be the same object
        passed to the DataLoader so ``set_epoch`` propagates correctly.
    local_rank : int or None, default=None
        CUDA device index for this process. Resolves to the ``LOCAL_RANK``
        environment variable when ``None``.
    **kwargs : dict
        Forwarded to ``PanelTrainer`` (``model``, ``optimizer``, ``loss_fn``,
        ``metrics``, ``callbacks``, ``grad_clip``, ``verbose``, ``strict``).
    """

    def __init__(
        self,
        *,
        train_sampler: DistributedSampler,
        local_rank: int | None = None,
        **kwargs: Any,
    ) -> None:
        if local_rank is None:
            local_rank = get_local_rank()

        kwargs["device"] = torch.device(f"cuda:{local_rank}")
        super().__init__(**kwargs)

        self._train_sampler = train_sampler
        self._local_rank = local_rank

        self.cfg.model = DistributedDataParallel(self.cfg.model, device_ids=[local_rank])

    def _on_train_epoch_start(self, epoch: int) -> None:
        super()._on_train_epoch_start(epoch)
        self._train_sampler.set_epoch(epoch)

    def _build_panel_batch(
        self,
        *,
        features: dict[str, Any],
        y: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> PanelBatch:
        """
        Build a panel batch, skipping the ``BasePanelModel`` isinstance check.

        Under DDP ``self.cfg.model`` is a ``DistributedDataParallel`` wrapper,
        not a direct ``BasePanelModel`` subclass. The underlying ``.module`` is
        the panel model; that check is performed here instead.
        """

        inner = getattr(self.cfg.model, "module", self.cfg.model)
        from dlecosys.models.panel_base import BasePanelModel

        if not isinstance(inner, BasePanelModel):
            raise TypeError(
                "PanelDistributedTrainer requires the wrapped model to subclass BasePanelModel."
            )
        if not isinstance(features, dict):
            raise TypeError(
                "PanelDistributedTrainer expects the feature payload to be a dict keyed by stream name."
            )
        return PanelBatch.from_training_batch(features=features, y=y, y_mask=y_mask)
