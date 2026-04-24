"""
Panel Preprocessing Package
---------------------------
Saveable preprocessing artifacts used by panel forecasting pipelines:
per-identifier standard scalers, per-column vocabulary tokenizers, and a
bundle that composes both into a single serializable artifact.

Classes
-------
Re-exported from submodules for convenience.
"""

from __future__ import annotations

from dlecosys.shared.preprocessing.panel.base import (
    ArtifactRecord,
    BasePreprocessorArtifact,
    BaseScaler,
    BaseTokenizer,
    decode_scalar,
    encode_scalar,
)
from dlecosys.shared.preprocessing.panel.bundle import PreprocessingBundle
from dlecosys.shared.preprocessing.panel.scalers import StandardScaler
from dlecosys.shared.preprocessing.panel.tokenizers import VocabularyTokenizer

__all__ = [
    "ArtifactRecord",
    "BasePreprocessorArtifact",
    "BaseScaler",
    "BaseTokenizer",
    "PreprocessingBundle",
    "StandardScaler",
    "VocabularyTokenizer",
    "decode_scalar",
    "encode_scalar",
]
