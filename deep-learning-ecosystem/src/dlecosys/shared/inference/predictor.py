"""
Predictor
---------
Batched inference pipeline for trained PyTorch models.

Classes
-------
Predictor
    Wraps a trained model with optional preprocessing and runs batched
    forward passes at inference time. Can be loaded directly from a
    ModelCheckpoint file.
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

__all__ = ["Predictor"]


class Predictor:
    """
    Batched inference wrapper for a trained PyTorch model.

    Applies an optional chain of preprocessors before running the model,
    then returns concatenated CPU predictions.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model. Moved to ``device`` and set to eval mode.
    device : str or torch.device, optional
        Device to run inference on. If None, tensors are not moved.
    preprocessors : list, optional
        Ordered sequence of objects that implement ``transform(X)``.
        Applied left-to-right before the model forward pass.

    Notes
    -----
    - ``model.eval()`` is called in ``__init__`` and again before each
      ``predict()`` call to guard against accidental training-mode state.
    - Input tensors are cast to the model's parameter dtype (typically
      float32) before inference.
    - Predictions are always returned on CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: Optional[Union[str, torch.device]] = None,
        preprocessors: Optional[List[Any]] = None,
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.model = model
        self.device = device
        self.preprocessors = list(preprocessors or [])

        if self.device is not None:
            self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        model: nn.Module,
        *,
        device: Optional[Union[str, torch.device]] = None,
        preprocessors: Optional[List[Any]] = None,
    ) -> "Predictor":
        """
        Load model weights from a ModelCheckpoint file.

        Parameters
        ----------
        path : str
            Path to a checkpoint saved by ModelCheckpoint (must contain
            a ``model_state_dict`` key).
        model : nn.Module
            Model instance with matching architecture. Weights are loaded
            in-place before the Predictor is constructed.
        device : str or torch.device, optional
        preprocessors : list, optional

        Returns
        -------
        Predictor
        """
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device=device, preprocessors=preprocessors)

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        *,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Run batched inference on X.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (N, ...)
            Input data. Preprocessors are applied in order before the
            model forward pass.
        batch_size : int, default 32
            Number of samples per forward pass.

        Returns
        -------
        torch.Tensor of shape (N, ...)
            Concatenated model outputs on CPU.
        """
        for preprocessor in self.preprocessors:
            X = preprocessor.transform(X)

        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(np.asarray(X, dtype=np.float32))

        # Cast to match the model's parameter dtype (usually float32).
        param = next(self.model.parameters(), None)
        if param is not None and X.dtype != param.dtype:
            X = X.to(param.dtype)

        self.model.eval()
        outputs: List[torch.Tensor] = []

        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                batch = X[start : start + batch_size]
                if self.device is not None:
                    batch = batch.to(self.device)
                outputs.append(self.model(batch).cpu())

        return torch.cat(outputs, dim=0)
