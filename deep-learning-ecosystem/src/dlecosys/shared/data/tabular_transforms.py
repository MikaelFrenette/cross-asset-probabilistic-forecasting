"""
Tabular Transforms
------------------
Shared kernel for fitting and applying per-column preprocessing
(integer tokenization for categorical columns, scaling for continuous columns)
to tabular feature tensors.

Used by preprocess.py (once per run, with persisted artifacts) and by the
per-fold tuning preprocessor (ephemeral, never persisted). Keeps the two
paths consistent as the preprocessing vocabulary grows.

Classes
-------
FittedTransforms
    Container for a fitted tokenizer-per-column dict and a fitted scaler
    (either may be absent depending on config).

Functions
---------
fit_transforms
    Fit tokenizers + scaler on a training X tensor.
apply_transforms
    Apply already-fit transforms to an X tensor. Returns a new tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch import Tensor

from dlecosys.shared.preprocessing import IntegerTokenizer, MinMaxScaler, StandardScaler

__all__ = ["FittedTransforms", "fit_transforms", "apply_transforms"]

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


@dataclass
class FittedTransforms:
    """Bundle of fitted per-column transforms.

    tokenizers : dict keyed on column index; each tokenizer has been fitted
                 on the training values of that column.
    scaler     : fitted scaler (StandardScaler or MinMaxScaler), or None if
                 scaling was disabled.
    cont_cols  : list of continuous column indices (complement of cat cols).
    """

    tokenizers: Dict[int, IntegerTokenizer] = field(default_factory=dict)
    scaler: Optional[object] = None
    cont_cols: List[int] = field(default_factory=list)


def fit_transforms(
    X_train: Tensor,
    *,
    categorical_cols: List[int],
    scaling: Optional[str],
) -> FittedTransforms:
    """
    Fit tokenizers and scaler on ``X_train`` only.

    Parameters
    ----------
    X_train : Tensor, shape (n_samples, n_features)
    categorical_cols : list of column indices to tokenize
    scaling : "standard", "minmax", or None (no scaling)
    """
    n_features = X_train.shape[1]
    cont_cols = [i for i in range(n_features) if i not in categorical_cols]

    tokenizers: Dict[int, IntegerTokenizer] = {}
    for col_idx in categorical_cols:
        tok = IntegerTokenizer()
        tok.fit(X_train[:, col_idx].long().numpy())
        tokenizers[col_idx] = tok

    scaler = None
    if scaling is not None and cont_cols:
        scaler = _SCALERS[scaling]()
        scaler.fit(X_train[:, cont_cols].numpy())

    return FittedTransforms(tokenizers=tokenizers, scaler=scaler, cont_cols=cont_cols)


def apply_transforms(X: Tensor, transforms: FittedTransforms) -> Tensor:
    """Apply fitted transforms to ``X``. Returns a modified clone (caller's tensor untouched)."""
    X = X.clone()
    for col_idx, tok in transforms.tokenizers.items():
        col = tok.transform(X[:, col_idx].long().numpy())
        X[:, col_idx] = torch.from_numpy(col).float()
    if transforms.scaler is not None and transforms.cont_cols:
        X[:, transforms.cont_cols] = transforms.scaler.transform(
            X[:, transforms.cont_cols]
        ).float()
    return X
