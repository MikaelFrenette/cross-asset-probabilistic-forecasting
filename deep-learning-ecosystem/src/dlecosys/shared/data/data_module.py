"""
Data Module
-----------
Abstract base class and configuration for dataset and DataLoader management.

Classes
-------
BaseDataModuleConfig
    Pydantic-validated DataLoader construction parameters shared across splits.
BaseDataModule
    Abstract interface for preparing datasets and producing DataLoaders for
    training, validation, and test splits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

__all__ = ["BaseDataModuleConfig", "BaseDataModule"]


class BaseDataModuleConfig(BaseModel):
    """
    DataLoader construction parameters shared across all splits.

    Parameters
    ----------
    batch_size : int, default 32
        Number of samples per batch.
    num_workers : int, default 0
        Number of subprocesses used for data loading. 0 means data is loaded
        in the main process.
    pin_memory : bool, default False
        If True, DataLoader copies tensors to pinned (page-locked) memory
        before returning them. Recommended when training on CUDA.
    drop_last : bool, default False
        Drop the last incomplete batch instead of yielding a smaller one.
    prefetch_factor : int, optional
        Number of batches to prefetch per worker. Only effective when
        num_workers > 0.
    """

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: Optional[int] = None


class BaseDataModule(ABC):
    """
    Abstract interface for dataset preparation and DataLoader construction.

    Subclasses implement ``setup()`` to initialize datasets (load, split,
    apply transforms) and ``train_dataloader()`` to expose the training split.
    Validation and test loaders are optional; they return None by default.

    The ``_make_loader`` helper centralizes DataLoader construction so
    split-specific methods only need to specify the dataset and shuffle policy.

    Parameters
    ----------
    config : BaseDataModuleConfig
        Validated DataLoader construction parameters.

    Notes
    -----
    Typical usage::

        dm = MyDataModule(config=BaseDataModuleConfig(batch_size=64))
        dm.setup()
        trainer.train(
            train_dataloader=dm.train_dataloader(),
            val_dataloader=dm.val_dataloader(),
            epochs=10,
        )
    """

    def __init__(self, config: BaseDataModuleConfig) -> None:
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Prepare datasets. Called once before accessing any dataloader."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the training split."""

    def val_dataloader(self) -> Optional[DataLoader]:
        """Return a DataLoader for the validation split, or None."""
        return None

    def test_dataloader(self) -> Optional[DataLoader]:
        """Return a DataLoader for the test split, or None."""
        return None

    def _make_loader(self, dataset: Dataset, *, shuffle: bool = False) -> DataLoader:
        """
        Build a DataLoader from this module's config.

        Parameters
        ----------
        dataset : Dataset
            PyTorch Dataset to wrap.
        shuffle : bool, default False
            Whether to shuffle at every epoch. Pass True for training splits.

        Returns
        -------
        DataLoader
        """
        cfg = self.config
        kwargs = dict(
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            shuffle=shuffle,
        )
        if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
            kwargs["prefetch_factor"] = cfg.prefetch_factor
        return DataLoader(dataset, **kwargs)
