"""
Training Inspectors
-------------------
Validation helpers for PyTorch models, optimizers, losses, and metrics used by
the training configuration layer.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from torch import nn
from torch.optim import Optimizer

__all__ = [
    "LossFunctionInspector",
    "MetricFunctionInspector",
    "ModelInspector",
    "OptimizerInspector",
    "TorchForwardInspector",
]


class TorchForwardInspector:
    """
    Validate objects exposing a PyTorch-style ``forward`` method.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """

    @staticmethod
    def _resolve_forward(fn: Any) -> Callable[..., Any]:
        """
        Resolve a callable interface from the provided object.

        Parameters
        ----------
        fn : Any
            Object expected to expose a callable interface.

        Returns
        -------
        collections.abc.Callable
            Bound or unbound callable used by the trainer.

        Raises
        ------
        TypeError
            If the object does not expose a compatible callable interface.
        """

        if hasattr(fn, "forward") and callable(getattr(fn, "forward")):
            return getattr(fn, "forward")
        if callable(fn):
            return fn
        raise TypeError("Object must be callable or expose a callable `forward(...)` method.")

    @staticmethod
    def _positional_params(callable_obj: Callable[..., Any]) -> list[inspect.Parameter]:
        """
        Extract positional parameters from a callable signature.

        Parameters
        ----------
        callable_obj : collections.abc.Callable
            Callable whose signature will be inspected.

        Returns
        -------
        list of inspect.Parameter
            Positional-only and positional-or-keyword parameters.
        """

        signature = inspect.signature(callable_obj)
        return [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

    @classmethod
    def validate(
        cls,
        fn: Any,
        *,
        name: str = "callable",
        min_positional: int = 2,
    ) -> None:
        """
        Validate a callable ``forward`` interface.

        Parameters
        ----------
        fn : Any
            Object exposing a callable ``forward`` method.
        name : str, default="callable"
            Human-readable object name used in validation messages.
        min_positional : int, default=2
            Minimum number of positional parameters required.
        Returns
        -------
        None
            This method raises when validation fails.

        Raises
        ------
        TypeError
            If the object does not expose a valid callable interface.
        ValueError
            If the callable signature does not satisfy the required contract.
        """

        resolved = cls._resolve_forward(fn)
        parameters = cls._positional_params(resolved)
        if len(parameters) < min_positional:
            raise ValueError(f"{name} must accept at least {min_positional} positional arguments.")


class LossFunctionInspector(TorchForwardInspector):
    """
    Validate a loss object compatible with the training loop.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """


class MetricFunctionInspector(TorchForwardInspector):
    """
    Validate a metric object compatible with the training loop.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """


class OptimizerInspector:
    """
    Validate optimizer objects used by the training loop.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """

    @staticmethod
    def validate(obj: Any, *, name: str = "optimizer") -> None:
        """
        Validate that the provided object is a PyTorch optimizer.

        Parameters
        ----------
        obj : Any
            Object expected to be a ``torch.optim.Optimizer``.
        name : str, default="optimizer"
            Human-readable object name used in validation messages.

        Returns
        -------
        None
            This method raises when validation fails.

        Raises
        ------
        TypeError
            If the provided object is not a PyTorch optimizer.
        """

        if not isinstance(obj, Optimizer):
            raise TypeError(f"{name} must be an instance of torch.optim.Optimizer.")


class ModelInspector:
    """
    Validate model objects used by the training loop.

    Parameters
    ----------
    None
        This class exposes only validation helpers.
    """

    @staticmethod
    def validate(obj: Any, *, name: str = "model") -> None:
        """
        Validate that the provided object is a PyTorch module.

        Parameters
        ----------
        obj : Any
            Object expected to be a ``torch.nn.Module``.
        name : str, default="model"
            Human-readable object name used in validation messages.

        Returns
        -------
        None
            This method raises when validation fails.

        Raises
        ------
        TypeError
            If the provided object is not a PyTorch module.
        """

        if not isinstance(obj, nn.Module):
            raise TypeError(f"{name} must be an instance of torch.nn.Module.")
