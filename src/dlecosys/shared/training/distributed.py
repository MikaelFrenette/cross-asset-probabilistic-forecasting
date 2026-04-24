"""
Distributed Trainer
--------------------
DistributedDataParallel wrapper around the supervised Trainer.

Classes
-------
DistributedTrainer
    Trainer subclass that wraps the model with DDP and seeds the
    DistributedSampler each epoch. All training loop logic is inherited.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from dlecosys.shared.training.process_group import get_local_rank
from dlecosys.shared.training.supervised import Trainer

__all__ = ["DistributedTrainer"]


class DistributedTrainer(Trainer):
    """
    Supervised trainer with DistributedDataParallel support.

    Wraps the model with DDP after parent initialisation and calls
    sampler.set_epoch(epoch) at the start of each training epoch so each
    rank sees a different data shard per epoch. All training loop, callback,
    metric, AMP, and gradient accumulation logic is fully inherited from
    Trainer / BaseTrainer.

    Parameters
    ----------
    train_sampler : DistributedSampler
        Sampler attached to the train DataLoader. Must be the same object
        passed to the DataLoader so that set_epoch propagates correctly.
    local_rank : int, optional
        CUDA device index for this process. Defaults to the LOCAL_RANK env var.
    **kwargs
        Forwarded to Trainer (model, optimizer, loss_fn, metrics, callbacks,
        amp, grad_accumulation_steps, grad_clip, verbose, ...).
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

        self.model = DistributedDataParallel(self.model, device_ids=[local_rank])

    def _on_train_epoch_start(self, epoch: int) -> None:
        super()._on_train_epoch_start(epoch)
        self._train_sampler.set_epoch(epoch)
