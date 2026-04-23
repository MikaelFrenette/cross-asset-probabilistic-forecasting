"""
Search Space Utilities
----------------------
Convert a flat ``{dotted.path: [choices]}`` search space to and from the
representation Optuna's samplers require, and apply suggested values into
a nested config dict.

Functions
---------
apply_suggestion
    Set a value in a nested config dict by dotted path.
suggest_values
    Iterate the search space and ask the trial to suggest a value for each entry.
to_hashable
    Recursively convert lists to tuples so they are hashable by Optuna.
from_hashable
    Recursively convert tuples back to lists for config round-trip.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["apply_suggestion", "suggest_values", "to_hashable", "from_hashable"]


def to_hashable(value: Any) -> Any:
    """Recursively convert lists to tuples (Optuna categorical choices must be hashable)."""
    if isinstance(value, list):
        return tuple(to_hashable(v) for v in value)
    return value


def from_hashable(value: Any) -> Any:
    """Recursively convert tuples back to lists for YAML/Pydantic round-trip."""
    if isinstance(value, tuple):
        return [from_hashable(v) for v in value]
    return value


def apply_suggestion(cfg_dict: Dict[str, Any], path: str, value: Any) -> None:
    """
    Set a value in a nested dict by dotted path, in place.

    Parameters
    ----------
    cfg_dict : dict
        Nested config dict (e.g. from ``PipelineConfig.model_dump()``).
    path : str
        Dotted path into the dict (e.g. ``"training.optimizer.lr"``).
    value : Any
        Value to assign at that path.

    Raises
    ------
    KeyError
        If an intermediate key does not exist.
    """
    parts = path.split(".")
    target = cfg_dict
    for part in parts[:-1]:
        if part not in target:
            raise KeyError(f"Search space path '{path}' does not exist in config (missing '{part}')")
        target = target[part]
    target[parts[-1]] = value


def suggest_values(trial, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Ask a trial to suggest a categorical value for each param in the search space.

    Lists (e.g. hidden_dims layer configs) are converted to tuples before being
    passed to Optuna and converted back to lists before being returned so they
    can be assigned back into the config.
    """
    suggestions: Dict[str, Any] = {}
    for path, choices in search_space.items():
        normalized = [to_hashable(c) for c in choices]
        chosen = trial.suggest_categorical(path, normalized)
        suggestions[path] = from_hashable(chosen)
    return suggestions
