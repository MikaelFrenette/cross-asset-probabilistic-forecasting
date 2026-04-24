"""
Scalers
-------
Feature scaling transforms for preprocessing pipelines and inference.

Classes
-------
BaseScaler
    Abstract base class for feature scalers. Defines the fit/transform
    interface. Artifact persistence is inherited from BaseArtifact.
StandardScaler
    Scales features to zero mean and unit variance (standard normal scaling).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..artifact import BaseArtifact

__all__ = ["BaseScaler", "MinMaxScaler", "StandardScaler"]


class BaseScaler(BaseArtifact):
    """
    Abstract base class for feature scalers.

    Defines the fit/transform interface. Save, load, and fit-state tracking
    are provided by BaseArtifact. Subclasses must implement fit, transform,
    inverse_transform, _get_state, and _set_state.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger for operational messages.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseScaler":
        """
        Compute scaling parameters from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BaseScaler
            Fitted scaler (enables method chaining).
        """

    @abstractmethod
    def transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply the fitted scaling transformation.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : same type and device as input
        """

    @abstractmethod
    def inverse_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reverse the scaling transformation.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        X_original : same type and device as input
        """

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to X, then return scaled X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : np.ndarray
        """
        return self.fit(X).transform(X)


class StandardScaler(BaseScaler):
    """
    Scale features to zero mean and unit variance.

    For each feature j: ``X_scaled[:, j] = (X[:, j] - mean_j) / std_j``

    Accepts numpy arrays for fit and both numpy arrays and PyTorch tensors for
    transform/inverse_transform. Tensors are returned on their original device.

    Parameters
    ----------
    eps : float, default=1e-8
        Floor applied to per-feature std before division. Prevents instability
        on constant or near-constant features.
    logger : logging.Logger, optional
        Logger for operational messages.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        Per-feature mean computed during fit.
    std_ : np.ndarray of shape (n_features,)
        Per-feature standard deviation computed during fit (floored to eps).
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X_scaled = scaler.fit_transform(X_train)
    >>> scaler.save("outputs/scalers/standard.pt")
    >>> scaler = StandardScaler.load("outputs/scalers/standard.pt")
    >>> X_test_scaled = scaler.transform(X_test_tensor)
    """

    def __init__(
        self,
        eps: float = 1e-8,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(logger=logger)
        self.eps = float(eps)
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute per-feature mean and std.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Must be 2D.

        Returns
        -------
        self : StandardScaler
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}.")

        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), self.eps)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True

        self.logger.debug(
            "%s fitted: %d samples, %d features | "
            "mean=[%.4f, %.4f] std=[%.4f, %.4f]",
            type(self).__name__,
            X.shape[0],
            self.n_features_in_,
            float(self.mean_.min()),
            float(self.mean_.max()),
            float(self.std_.min()),
            float(self.std_.max()),
        )
        return self

    def transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply standard scaling: ``(X - mean) / std``.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        X_scaled : same type and device as input
        """
        self._check_fitted()
        X_np, is_tensor, device = _to_numpy(X)
        self._validate_n_features(X_np)
        return _restore_type((X_np - self.mean_) / self.std_, is_tensor, device)

    def inverse_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Reverse standard scaling: ``X * std + mean``.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (n_samples, n_features)

        Returns
        -------
        X_original : same type and device as input
        """
        self._check_fitted()
        X_np, is_tensor, device = _to_numpy(X)
        self._validate_n_features(X_np)
        return _restore_type(X_np * self.std_ + self.mean_, is_tensor, device)

    def _validate_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features; got {X.shape[1]}."
            )

    def _get_state(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "mean_": self.mean_,
            "std_": self.std_,
            "n_features_in_": self.n_features_in_,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.eps = state["eps"]
        self.mean_ = state["mean_"]
        self.std_ = state["std_"]
        self.n_features_in_ = state["n_features_in_"]
        self._is_fitted = True


class MinMaxScaler(BaseScaler):
    """
    Scale features to the [0, 1] range.

    For each feature j: ``X_scaled[:, j] = (X[:, j] - min_j) / (max_j - min_j)``

    Parameters
    ----------
    eps : float, default=1e-8
        Floor applied to per-feature range before division. Prevents instability
        on constant or near-constant features.
    logger : logging.Logger, optional
    """

    def __init__(
        self,
        eps: float = 1e-8,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(logger=logger)
        self.eps = float(eps)
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}.")
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        self.n_features_in_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self._check_fitted()
        X_np, is_tensor, device = _to_numpy(X)
        self._validate_n_features(X_np)
        scale = np.maximum(self.max_ - self.min_, self.eps)
        return _restore_type((X_np - self.min_) / scale, is_tensor, device)

    def inverse_transform(
        self, X: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        self._check_fitted()
        X_np, is_tensor, device = _to_numpy(X)
        self._validate_n_features(X_np)
        scale = np.maximum(self.max_ - self.min_, self.eps)
        return _restore_type(X_np * scale + self.min_, is_tensor, device)

    def _validate_n_features(self, X: np.ndarray) -> None:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features; got {X.shape[1]}."
            )

    def _get_state(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "min_": self.min_,
            "max_": self.max_,
            "n_features_in_": self.n_features_in_,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        self.eps = state["eps"]
        self.min_ = state["min_"]
        self.max_ = state["max_"]
        self.n_features_in_ = state["n_features_in_"]
        self._is_fitted = True


# ---------------------------------------------------------------------------
# Module-level helpers — not part of the public API
# ---------------------------------------------------------------------------

def _to_numpy(
    X: Union[np.ndarray, torch.Tensor],
) -> Tuple[np.ndarray, bool, Optional[torch.device]]:
    """Normalize input to float64 numpy, returning metadata to restore the original type."""
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy().astype(np.float64), True, X.device
    return np.asarray(X, dtype=np.float64), False, None


def _restore_type(
    X: np.ndarray,
    is_tensor: bool,
    device: Optional[torch.device],
) -> Union[np.ndarray, torch.Tensor]:
    """Return a tensor on the original device if input was a tensor, else numpy."""
    if is_tensor:
        return torch.from_numpy(X).to(device)
    return X
