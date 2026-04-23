"""
Model Factory
-------------
Registry and factory for instantiating nn.Module models by name from config dicts.

Functions
---------
register
    Class decorator that registers an nn.Module in the global model registry.

Classes
-------
ModelFactory
    Builds registered models from a name and a parameter dictionary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

import torch.nn as nn

__all__ = ["ModelFactory", "register"]

_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register(name: str):
    """
    Register an ``nn.Module`` subclass under ``name`` in the global model registry.

    The decorated class must define:

    - ``config_class``: a :class:`~dlecosys.shared.models.ModelConfig` subclass
    - ``from_config(cls, config)``: classmethod returning an ``nn.Module`` instance

    Parameters
    ----------
    name : str
        Registry key used to look up the model via :meth:`ModelFactory.build`.

    Raises
    ------
    ValueError
        If ``name`` is already taken in the registry.
    TypeError
        If the class is missing ``config_class`` or ``from_config``.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _REGISTRY:
            raise ValueError(
                f"A model named {name!r} is already registered. "
                f"Choose a different name or remove the existing registration."
            )
        if not hasattr(cls, "config_class"):
            raise TypeError(
                f"{cls.__name__} must define a 'config_class' class attribute "
                f"pointing to its ModelConfig subclass."
            )
        if not hasattr(cls, "from_config"):
            raise TypeError(
                f"{cls.__name__} must implement a 'from_config' classmethod."
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


class ModelFactory:
    """
    Instantiates registered models by name from a parameter dictionary.

    The factory resolves the model class from the registry, validates the
    parameters against its ``config_class``, then delegates construction to
    ``from_config``.
    """

    @staticmethod
    def build(name: str, params: Dict[str, Any]) -> nn.Module:
        """
        Build a model by registry name.

        Parameters
        ----------
        name : str
            Registry key for the model class.
        params : dict
            Keyword arguments forwarded to the model's ``config_class``.

        Returns
        -------
        nn.Module

        Raises
        ------
        KeyError
            If ``name`` is not found in the registry.
        pydantic.ValidationError
            If ``params`` do not satisfy the model's ``config_class`` schema.
        """
        if name not in _REGISTRY:
            available = ModelFactory.available()
            raise KeyError(
                f"No model named {name!r} in the registry. "
                f"Available: {available}"
            )
        cls = _REGISTRY[name]
        config = cls.config_class(**params)
        return cls.from_config(config)

    @staticmethod
    def available() -> List[str]:
        """Return a sorted list of registered model names."""
        return sorted(_REGISTRY)
