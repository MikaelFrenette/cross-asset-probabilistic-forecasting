"""
Workflow Package
----------------
Repository-level extraction and training entrypoints built from typed configs.
"""

from density_model.workflows.extraction import extract_features_from_config
from density_model.workflows.training import train_from_config

__all__ = ["extract_features_from_config", "train_from_config"]
