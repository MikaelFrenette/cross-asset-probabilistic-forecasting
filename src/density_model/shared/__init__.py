"""
Shared Utilities
----------------
Cross-domain utilities and helpers shared across density_model subpackages.

Classes
-------
BaseArtifact
    Abstract mixin providing save/load persistence and fit-state tracking
    for all fitted objects (scalers, tokenizers, and future artifact types).
ModelConfig
    Pydantic base class all model-specific configs must subclass.
ModelFactory
    Builds registered models from a name and a parameter dictionary.

Functions
---------
seed_everything
    Seeds Python, NumPy, PyTorch, and CUDA RNGs from a single integer seed.
register
    Class decorator that registers an nn.Module in the global model registry.
"""

from density_model.shared.artifact import BaseArtifact
from density_model.shared.models import ModelConfig, ModelFactory, register
from density_model.shared.reproducibility import seed_everything

__all__ = ["BaseArtifact", "ModelConfig", "ModelFactory", "register", "seed_everything"]
