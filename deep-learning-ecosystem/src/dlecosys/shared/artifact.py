"""
Artifact Persistence
--------------------
Base class for persistable fitted objects in the dlecosys ecosystem.

Classes
-------
BaseArtifact
    Abstract mixin providing save/load artifact persistence and fit-state
    tracking. Combine with a domain-specific abstract base (e.g. BaseScaler,
    BaseTokenizer) to gain disk persistence without duplicating serialization
    logic across preprocessing classes.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

__all__ = ["BaseArtifact"]


class BaseArtifact(ABC):
    """
    Abstract mixin for fitted objects that can be saved and loaded from disk.

    Handles the full persistence contract: fit-state tracking, class-identity
    validation on load, directory creation on save, and logger injection.
    Subclasses declare their serializable state via _get_state/_set_state.

    Parameters
    ----------
    logger : logging.Logger, optional
        Logger for operational messages. Defaults to the logger for the
        concrete subclass's module, so log lines are attributed correctly
        regardless of where BaseArtifact lives in the hierarchy.

    Attributes
    ----------
    _is_fitted : bool
        True after the subclass has completed a successful fit.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(type(self).__module__)
        self._is_fitted: bool = False

    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """Return a serializable dict of all fitted parameters."""

    @abstractmethod
    def _set_state(self, state: Dict[str, Any]) -> None:
        """Restore all fitted parameters from a state dict produced by _get_state."""

    def save(self, path: str) -> None:
        """
        Persist the fitted artifact to disk.

        Parameters
        ----------
        path : str
            Destination file path (e.g. "outputs/scalers/standard.pt").
            Parent directories are created automatically.

        Raises
        ------
        RuntimeError
            If called before the artifact has been fitted.
        """
        self._check_fitted("save")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save({"class": type(self).__name__, "state": self._get_state()}, path)
        self.logger.info("Saved %s to '%s'.", type(self).__name__, path)

    @classmethod
    def load(cls, path: str, logger: Optional[logging.Logger] = None) -> "BaseArtifact":
        """
        Load a fitted artifact from disk.

        Parameters
        ----------
        path : str
            File path of the saved artifact.
        logger : logging.Logger, optional
            Logger to attach to the loaded instance.

        Returns
        -------
        artifact : instance of cls
            Fitted artifact ready for use.

        Raises
        ------
        TypeError
            If the artifact was saved by a different class.
        """
        payload = torch.load(path, weights_only=False)
        saved_class = payload.get("class", "")
        if saved_class != cls.__name__:
            raise TypeError(
                f"Artifact was saved by '{saved_class}', cannot load into '{cls.__name__}'."
            )
        obj = object.__new__(cls)
        BaseArtifact.__init__(obj, logger=logger)
        obj._set_state(payload["state"])
        obj.logger.info("Loaded %s from '%s'.", cls.__name__, path)
        return obj

    def _check_fitted(self, method: str = "transform") -> None:
        """Raise RuntimeError if the artifact has not been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling '{method}'. "
                "Call fit() first."
            )
