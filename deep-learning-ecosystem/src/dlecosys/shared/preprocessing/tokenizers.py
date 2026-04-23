"""
Tokenizers
----------
Vocabulary-based tokenizers for categorical and integer features.

Classes
-------
BaseTokenizer
    Abstract base class defining the fit/transform interface and vocabulary
    management. Artifact persistence is inherited from BaseArtifact.
IntegerTokenizer
    Builds a vocabulary from integer values and maps them to contiguous token
    IDs, sorted numerically for deterministic assignment.
StringTokenizer
    Builds a vocabulary from string values and maps them to contiguous token
    IDs, sorted alphabetically for deterministic assignment.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..artifact import BaseArtifact

__all__ = ["BaseTokenizer", "IntegerTokenizer", "StringTokenizer"]

_UNK = "<unk>"
_UNK_ID = 0


class BaseTokenizer(BaseArtifact):
    """
    Abstract base class for vocabulary-based tokenizers.

    Manages vocabulary construction, token lookup, and unknown-token handling.
    Save, load, and fit-state tracking are provided by BaseArtifact. Subclasses
    implement fit and state serialization; transform/inverse_transform are
    concrete here since the lookup logic is identical across all categorical
    tokenizers.

    Parameters
    ----------
    handle_unknown : {"error", "unk"}, default "error"
        How to handle values not seen during fit.
        - "error": raises KeyError on unseen values.
        - "unk": maps unseen values to a reserved <unk> token (ID 0).
          All fitted vocab IDs are shifted by 1 to accommodate it.
    logger : logging.Logger, optional
        Logger for operational messages.

    Attributes
    ----------
    _vocab : dict
        Mapping from original value to token ID.
    _vocab_inv : dict
        Mapping from token ID to original value.
    """

    def __init__(
        self,
        handle_unknown: str = "error",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if handle_unknown not in {"error", "unk"}:
            raise ValueError(
                f"handle_unknown must be 'error' or 'unk'; got {handle_unknown!r}."
            )
        super().__init__(logger=logger)
        self.handle_unknown = handle_unknown
        self._vocab: Dict[Any, int] = {}
        self._vocab_inv: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseTokenizer":
        """
        Build vocabulary from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Training values. All unique values are added to the vocabulary.

        Returns
        -------
        self : BaseTokenizer
            Fitted tokenizer (enables method chaining).
        """

    # ------------------------------------------------------------------
    # Concrete transform logic — shared by all categorical tokenizers
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map values to token IDs.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
            Values to encode. Must have been seen during fit unless
            handle_unknown='unk'.

        Returns
        -------
        X_ids : np.ndarray of int64, same shape as input
        """
        self._check_fitted()
        X = np.asarray(X)
        shape = X.shape
        ids = np.array([self._lookup(v) for v in X.ravel()], dtype=np.int64)
        return ids.reshape(shape)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map token IDs back to original values.

        Parameters
        ----------
        X : array-like of int64, shape (n_samples,) or (n_samples, n_features)
            Token IDs to decode.

        Returns
        -------
        X_decoded : np.ndarray, same shape as input
            Original values. Unknown IDs decode to None.
        """
        self._check_fitted()
        X = np.asarray(X)
        shape = X.shape
        vals = np.array(
            [self._vocab_inv.get(int(v)) for v in X.ravel()],
            dtype=object,
        )
        return vals.reshape(shape)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to X, then return token IDs for X.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        X_ids : np.ndarray of int64
        """
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including <unk> if handle_unknown='unk'."""
        return len(self._vocab)

    @property
    def vocab_(self) -> Dict[Any, int]:
        """Shallow copy of the value → ID mapping."""
        return dict(self._vocab)

    def _build_vocab(self, unique_values, sort_key=None) -> None:
        """
        Assign contiguous IDs to unique_values.

        When handle_unknown='unk', <unk> is reserved at ID 0 and all other
        IDs start at 1. When handle_unknown='error', IDs start at 0.
        """
        sorted_vals = sorted(unique_values, key=sort_key)
        if self.handle_unknown == "unk":
            self._vocab = {_UNK: _UNK_ID}
            offset = 1
        else:
            self._vocab = {}
            offset = 0
        for i, v in enumerate(sorted_vals):
            self._vocab[v] = i + offset
        self._vocab_inv = {v: k for k, v in self._vocab.items()}

    def _lookup(self, value: Any) -> int:
        """Return the token ID for value, applying unknown handling if needed."""
        token_id = self._vocab.get(value)
        if token_id is not None:
            return token_id
        if self.handle_unknown == "unk":
            return _UNK_ID
        raise KeyError(
            f"Unknown token {value!r}. Either include it in training data "
            "or set handle_unknown='unk'."
        )


class IntegerTokenizer(BaseTokenizer):
    """
    Build a vocabulary from integer values and map them to token IDs.

    Vocabulary IDs are assigned in ascending numerical order of the original
    values, making the mapping deterministic and human-readable.

    Parameters
    ----------
    handle_unknown : {"error", "unk"}, default "error"
        How to handle integers not seen during fit.
    logger : logging.Logger, optional
        Logger for operational messages.

    Attributes
    ----------
    vocab_size : int
        Number of tokens (including <unk> if handle_unknown='unk').

    Examples
    --------
    >>> tok = IntegerTokenizer(handle_unknown="unk")
    >>> ids = tok.fit_transform(np.array([10, 20, 30, 10]))
    >>> tok.save("outputs/tokenizers/item_ids.pt")
    >>> tok = IntegerTokenizer.load("outputs/tokenizers/item_ids.pt")
    >>> tok.transform(np.array([10, 99]))  # 99 → <unk> ID
    """

    def fit(self, X: np.ndarray) -> "IntegerTokenizer":
        """
        Build vocabulary from integer values.

        Parameters
        ----------
        X : array-like of int, shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        self : IntegerTokenizer
        """
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.integer):
            raise TypeError(
                f"IntegerTokenizer expects integer input; got dtype {X.dtype}."
            )
        unique_vals = set(int(v) for v in X.ravel())
        self._build_vocab(unique_vals)
        self._is_fitted = True
        self.logger.debug(
            "%s fitted: %d unique integers → vocab_size=%d.",
            type(self).__name__,
            len(unique_vals),
            self.vocab_size,
        )
        return self

    def _get_state(self) -> Dict[str, Any]:
        return {
            "handle_unknown": self.handle_unknown,
            "_vocab": self._vocab,
            "_vocab_inv": self._vocab_inv,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.handle_unknown = state["handle_unknown"]
        self._vocab = state["_vocab"]
        self._vocab_inv = state["_vocab_inv"]
        self._is_fitted = True


class StringTokenizer(BaseTokenizer):
    """
    Build a vocabulary from string values and map them to token IDs.

    Vocabulary IDs are assigned in ascending alphabetical order of the original
    values, making the mapping deterministic.

    Parameters
    ----------
    handle_unknown : {"error", "unk"}, default "error"
        How to handle strings not seen during fit.
    logger : logging.Logger, optional
        Logger for operational messages.

    Attributes
    ----------
    vocab_size : int
        Number of tokens (including <unk> if handle_unknown='unk').

    Examples
    --------
    >>> tok = StringTokenizer(handle_unknown="unk")
    >>> ids = tok.fit_transform(np.array(["cat", "dog", "cat"]))
    >>> tok.save("outputs/tokenizers/categories.pt")
    >>> tok = StringTokenizer.load("outputs/tokenizers/categories.pt")
    >>> tok.transform(np.array(["cat", "bird"]))  # "bird" → <unk> ID
    """

    def fit(self, X: np.ndarray) -> "StringTokenizer":
        """
        Build vocabulary from string values.

        Parameters
        ----------
        X : array-like of str, shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        self : StringTokenizer
        """
        X = np.asarray(X)
        if X.dtype.kind not in {"U", "O"}:
            raise TypeError(
                f"StringTokenizer expects string input; got dtype {X.dtype}."
            )
        unique_vals = set(str(v) for v in X.ravel())
        self._build_vocab(unique_vals)
        self._is_fitted = True
        self.logger.debug(
            "%s fitted: %d unique strings → vocab_size=%d.",
            type(self).__name__,
            len(unique_vals),
            self.vocab_size,
        )
        return self

    def _get_state(self) -> Dict[str, Any]:
        return {
            "handle_unknown": self.handle_unknown,
            "_vocab": self._vocab,
            "_vocab_inv": self._vocab_inv,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.handle_unknown = state["handle_unknown"]
        self._vocab = state["_vocab"]
        self._vocab_inv = state["_vocab_inv"]
        self._is_fitted = True
