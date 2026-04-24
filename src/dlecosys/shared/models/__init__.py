"""
Model Factory
-------------
Registry and factory for building nn.Module models by name from config dicts.

Classes
-------
ModelConfig
    Pydantic base class all model-specific configs must subclass.
ModelFactory
    Builds registered models from a name and a parameter dictionary.

Functions
---------
register
    Class decorator that registers an nn.Module in the global model registry.
"""

from dlecosys.shared.models.base import ModelConfig
from dlecosys.shared.models.factory import ModelFactory, register

__all__ = ["ModelConfig", "ModelFactory", "register"]
