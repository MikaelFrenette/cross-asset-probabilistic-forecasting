"""
Shared Utilities
----------------
Cross-domain utilities and helpers shared across dlecosys subpackages.

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

from dlecosys.shared.artifact import BaseArtifact
from dlecosys.shared.models import ModelConfig, ModelFactory, register
from dlecosys.shared.reproducibility import seed_everything

__all__ = ["BaseArtifact", "ModelConfig", "ModelFactory", "register", "seed_everything"]
