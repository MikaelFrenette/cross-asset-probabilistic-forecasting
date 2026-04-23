"""
Fold Preprocessing
------------------
Ephemeral per-fold tokenization + scaling for tuning trials.

The tuning objective receives raw (train_ds, val_ds) pairs from a Splitter,
then calls ``preprocess_fold`` to fit fresh transforms on the fold's train
data only and apply them to both datasets. The scaler and tokenizers are
garbage-collected at fold end — nothing is written to disk.

Functions
---------
preprocess_fold
    Fit + apply tabular transforms for a single fold.
"""

from __future__ import annotations

from typing import Tuple

from torch.utils.data import TensorDataset

from dlecosys.shared.config.schema import DataSection
from dlecosys.shared.data.tabular_transforms import apply_transforms, fit_transforms

__all__ = ["preprocess_fold"]


def preprocess_fold(
    data_cfg: DataSection,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Fit ephemeral transforms on the fold's train data and apply to both splits.

    No artifacts are persisted — the scaler and tokenizers fall out of scope
    when this function returns.
    """
    X_train, y_train = train_ds.tensors
    X_val, y_val = val_ds.tensors

    transforms = fit_transforms(
        X_train,
        categorical_cols=data_cfg.categorical_cols,
        scaling=data_cfg.scaling,
    )

    X_train_t = apply_transforms(X_train, transforms)
    X_val_t = apply_transforms(X_val, transforms)

    return TensorDataset(X_train_t, y_train), TensorDataset(X_val_t, y_val)
