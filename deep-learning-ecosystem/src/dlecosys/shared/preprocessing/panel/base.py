"""
Panel Preprocessing Base
------------------------
Abstract saveable artifact base classes for extensible panel preprocessing
components with shared JSON save/load semantics.

Classes
-------
ArtifactRecord
    Pydantic envelope persisted to disk for every artifact (type + state).

BasePreprocessorArtifact
    Abstract saveable artifact with subclass registry keyed by
    ``artifact_type``.

BaseScaler
    Abstract fit/transform/inverse_transform scaler artifact.

BaseTokenizer
    Abstract fit/transform/inverse_transform tokenizer artifact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ArtifactRecord",
    "BasePreprocessorArtifact",
    "BaseScaler",
    "BaseTokenizer",
    "decode_scalar",
    "encode_scalar",
]


class ArtifactRecord(BaseModel):
    """Serialized artifact record for preprocessing components."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    state: dict[str, Any]


class BasePreprocessorArtifact(ABC):
    """Abstract base class for saveable preprocessing artifacts."""

    artifact_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(BasePreprocessorArtifact, "_registry"):
            BasePreprocessorArtifact._registry = {}
        artifact_type = getattr(cls, "artifact_type", None)
        if artifact_type:
            BasePreprocessorArtifact._registry[artifact_type] = cls

    def save(self, path: str | Path) -> None:
        """Save the artifact state to disk."""

        record = ArtifactRecord(artifact_type=self.artifact_type, state=self._serialize_state())
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BasePreprocessorArtifact:
        """Load an artifact from disk."""

        record = ArtifactRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))
        artifact_cls = BasePreprocessorArtifact._registry.get(record.artifact_type)
        if artifact_cls is None:
            raise ValueError(f"Unknown artifact type: {record.artifact_type!r}")
        if cls is not BasePreprocessorArtifact and artifact_cls is not cls:
            raise TypeError(
                f"Artifact type mismatch: requested {cls.__name__}, "
                f"found {artifact_cls.__name__} in serialized state."
            )
        return artifact_cls._deserialize_state(record.state)

    @abstractmethod
    def _serialize_state(self) -> dict[str, Any]:
        """Serialize artifact-specific state."""

    @classmethod
    @abstractmethod
    def _deserialize_state(cls, state: dict[str, Any]) -> BasePreprocessorArtifact:
        """Reconstruct an artifact from serialized state."""


def encode_scalar(value: Any) -> dict[str, Any]:
    """Serialize a scalar value with explicit type information."""

    if value is None:
        return {"type": "none", "value": None}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise TypeError(f"Unsupported scalar type for serialization: {type(value).__name__}")


def decode_scalar(payload: dict[str, Any]) -> Any:
    """Decode a scalar value serialized by :func:`encode_scalar`."""

    payload_type = payload["type"]
    value = payload["value"]
    if payload_type == "none":
        return None
    if payload_type == "bool":
        return bool(value)
    if payload_type == "int":
        return int(value)
    if payload_type == "float":
        return float(value)
    if payload_type == "str":
        return str(value)
    raise ValueError(f"Unsupported serialized scalar type: {payload_type!r}")


class BaseScaler(BasePreprocessorArtifact, ABC):
    """Abstract base class for fit-transform scaling artifacts."""

    @abstractmethod
    def fit(self, values: Any) -> BaseScaler:
        """Fit the scaler on input values."""

    @abstractmethod
    def transform(self, values: Any) -> Any:
        """Transform input values using fitted scaling statistics."""

    @abstractmethod
    def inverse_transform(self, values: Any) -> Any:
        """Invert a scaling transformation."""


class BaseTokenizer(BasePreprocessorArtifact, ABC):
    """Abstract base class for categorical tokenization artifacts."""

    @abstractmethod
    def fit(self, values: Any) -> BaseTokenizer:
        """Fit the tokenizer vocabulary on input values."""

    @abstractmethod
    def transform(self, values: Any) -> Any:
        """Transform categorical values into token ids."""

    @abstractmethod
    def inverse_transform(self, values: Any) -> Any:
        """Map token ids back to categorical values."""
