"""
Data Utilities
--------------
DataModule abstraction, synthetic data generation, and pipeline data validation.

Classes
-------
BaseDataModule
    Abstract interface for setting up datasets and producing DataLoaders for
    training, validation, and test splits.
BaseDataModuleConfig
    Pydantic-validated DataLoader construction parameters (batch size, workers,
    pin_memory, drop_last, prefetch_factor).

Functions
---------
make_regression_splits
    Generate a synthetic linear regression dataset split into train/val/test.
make_classification_splits
    Generate a synthetic cluster-based classification dataset split into train/val/test.
validate_split
    Raise if a single (X, y) split contains NaN, Inf, wrong shape, or non-float X.
validate_splits
    Apply validate_split to every split in a dict.
"""

from dlecosys.shared.data.data_module import BaseDataModule, BaseDataModuleConfig
from dlecosys.shared.data.fold_preprocessing import preprocess_fold
from dlecosys.shared.data.sources import load_tabular
from dlecosys.shared.data.splitters import (
    BaseSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    StratifiedKFoldSplitter,
    build_splitter,
)
from dlecosys.shared.data.synthetic import make_classification_splits, make_regression_splits, split_tensors
from dlecosys.shared.data.tabular_transforms import FittedTransforms, apply_transforms, fit_transforms
from dlecosys.shared.data.validation import validate_split, validate_splits

__all__ = [
    "BaseDataModule",
    "BaseDataModuleConfig",
    "make_regression_splits",
    "make_classification_splits",
    "split_tensors",
    "validate_split",
    "validate_splits",
    "BaseSplitter",
    "HoldoutSplitter",
    "KFoldSplitter",
    "StratifiedKFoldSplitter",
    "build_splitter",
    "FittedTransforms",
    "fit_transforms",
    "apply_transforms",
    "preprocess_fold",
    "load_tabular",
]
