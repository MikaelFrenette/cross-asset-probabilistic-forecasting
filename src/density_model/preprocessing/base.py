"""
Preprocessing Base Classes
--------------------------
Abstract artifact classes for extensible preprocessing components with shared
save and load behavior.
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
    """
    Serialized artifact record for preprocessing components.

    Parameters
    ----------
    artifact_type : str
        Registered public artifact type name.
    state : dict of str to Any
        Artifact-specific serialized state.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_type: str
    state: dict[str, Any]


class BasePreprocessorArtifact(ABC):
    """
    Abstract base class for saveable preprocessing artifacts.

    Parameters
    ----------
    None
        Subclasses manage their own fitting state.
    """

    artifact_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Register artifact subclasses by public type name.

        Parameters
        ----------
        **kwargs : Any
            Additional subclass construction arguments.

        Returns
        -------
        None
            This method updates the artifact registry.
        """

        super().__init_subclass__(**kwargs)
        if not hasattr(BasePreprocessorArtifact, "_registry"):
            BasePreprocessorArtifact._registry = {}
        artifact_type = getattr(cls, "artifact_type", None)
        if artifact_type:
            BasePreprocessorArtifact._registry[artifact_type] = cls

    def save(self, path: str | Path) -> None:
        """
        Save the artifact state to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Output JSON file path.

        Returns
        -------
        None
            This method writes serialized artifact state.
        """

        record = ArtifactRecord(artifact_type=self.artifact_type, state=self._serialize_state())
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(record.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BasePreprocessorArtifact:
        """
        Load an artifact from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Input JSON file path.

        Returns
        -------
        BasePreprocessorArtifact
            Loaded preprocessing artifact.

        Raises
        ------
        ValueError
            If the serialized artifact type is unknown.
        TypeError
            If the current class is incompatible with the serialized artifact type.
        """

        record = ArtifactRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))
        artifact_cls = BasePreprocessorArtifact._registry.get(record.artifact_type)
        if artifact_cls is None:
            raise ValueError(f"Unknown artifact type: {record.artifact_type!r}")
        if cls is not BasePreprocessorArtifact and artifact_cls is not cls:
            raise TypeError(
                f"Artifact type mismatch: requested {cls.__name__}, found {artifact_cls.__name__} in serialized state."
            )
        return artifact_cls._deserialize_state(record.state)

    @abstractmethod
    def _serialize_state(self) -> dict[str, Any]:
        """
        Serialize artifact-specific state.

        Returns
        -------
        dict of str to Any
            Serialized artifact state.
        """

    @classmethod
    @abstractmethod
    def _deserialize_state(cls, state: dict[str, Any]) -> BasePreprocessorArtifact:
        """
        Reconstruct an artifact from serialized state.

        Parameters
        ----------
        state : dict of str to Any
            Serialized artifact state.

        Returns
        -------
        BasePreprocessorArtifact
            Reconstructed artifact.
        """


def encode_scalar(value: Any) -> dict[str, Any]:
    """
    Serialize a scalar value with explicit type information.

    Parameters
    ----------
    value : Any
        Scalar value to serialize.

    Returns
    -------
    dict of str to Any
        Serialized scalar payload.

    Raises
    ------
    TypeError
        If the value type is unsupported.
    """

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
    """
    Decode a scalar value serialized by :func:`encode_scalar`.

    Parameters
    ----------
    payload : dict of str to Any
        Serialized scalar payload.

    Returns
    -------
    Any
        Decoded scalar value.

    Raises
    ------
    ValueError
        If the payload type tag is unsupported.
    """

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
    """
    Abstract base class for fit-transform scaling artifacts.

    Parameters
    ----------
    None
        Subclasses manage scaling statistics.
    """

    @abstractmethod
    def fit(self, values: Any) -> BaseScaler:
        """
        Fit the scaler on input values.

        Parameters
        ----------
        values : Any
            Input values used to estimate scaling statistics.

        Returns
        -------
        BaseScaler
            Fitted scaler instance.
        """

    @abstractmethod
    def transform(self, values: Any) -> Any:
        """
        Transform input values using fitted scaling statistics.

        Parameters
        ----------
        values : Any
            Input values to transform.

        Returns
        -------
        Any
            Scaled output values.
        """

    @abstractmethod
    def inverse_transform(self, values: Any) -> Any:
        """
        Invert a scaling transformation.

        Parameters
        ----------
        values : Any
            Scaled values to invert.

        Returns
        -------
        Any
            Values mapped back to the original scale.
        """


class BaseTokenizer(BasePreprocessorArtifact, ABC):
    """
    Abstract base class for categorical tokenization artifacts.

    Parameters
    ----------
    None
        Subclasses manage token vocabularies and mappings.
    """

    @abstractmethod
    def fit(self, values: Any) -> BaseTokenizer:
        """
        Fit the tokenizer vocabulary on input values.

        Parameters
        ----------
        values : Any
            Input categorical values used to build the vocabulary.

        Returns
        -------
        BaseTokenizer
            Fitted tokenizer instance.
        """

    @abstractmethod
    def transform(self, values: Any) -> Any:
        """
        Transform categorical values into token ids.

        Parameters
        ----------
        values : Any
            Input categorical values to tokenize.

        Returns
        -------
        Any
            Tokenized output values.
        """

    @abstractmethod
    def inverse_transform(self, values: Any) -> Any:
        """
        Map token ids back to categorical values.

        Parameters
        ----------
        values : Any
            Token ids to decode.

        Returns
        -------
        Any
            Decoded categorical values.
        """
