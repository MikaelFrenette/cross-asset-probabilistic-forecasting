"""
Config Loader
-------------
Loads and validates YAML pipeline configs with single-level base inheritance.

Functions
---------
load_config
    Load a YAML config, optionally merging with a base config, and return
    a validated PipelineConfig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from dlecosys.shared.config.schema import PipelineConfig

__all__ = ["load_config"]


class ConfigError(ValueError):
    """Raised when the config file or its inheritance chain is invalid."""


def load_config(path: str | Path) -> PipelineConfig:
    """
    Load and validate a pipeline config YAML.

    If the config contains a ``base`` key, the referenced file is loaded
    and deep-merged with the current config (current values override base).
    Inheritance is capped at 1 level — a base config must not itself define
    a ``base`` key.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    PipelineConfig

    Raises
    ------
    ConfigError
        If the inheritance chain exceeds 1 level.
    FileNotFoundError
        If the config file or its declared base does not exist.
    pydantic.ValidationError
        If the merged config fails schema validation.
    """
    path = Path(path).resolve()
    raw = _load_yaml(path)
    base_path_str = raw.pop("base", None)

    if base_path_str is not None:
        base_path = (path.parent / base_path_str).resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")

        base_raw = _load_yaml(base_path)

        if base_raw.get("base") is not None:
            raise ConfigError(
                f"Config inheritance cannot exceed 1 level. "
                f"'{base_path}' itself declares a 'base' key. "
                f"Flatten your config hierarchy."
            )
        base_raw.pop("base", None)
        merged = _deep_merge(base_raw, raw)
    else:
        merged = raw

    return PipelineConfig(**merged)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base; override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
