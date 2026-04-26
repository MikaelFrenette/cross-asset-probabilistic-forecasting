"""
Shared Utilities
----------------
Cross-domain utilities shared across density_model subpackages.

Classes
-------
ModelConfig
    Pydantic base class all model-specific configs must subclass.
ModelFactory
    Builds registered models from a name and a parameter dictionary.

Functions
---------
register
    Class decorator that registers an ``nn.Module`` in the global model registry.
"""

from density_model.shared.models import ModelConfig, ModelFactory, register

__all__ = ["ModelConfig", "ModelFactory", "register"]
