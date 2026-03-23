"""
Training Manifest
-----------------
Frozen manifest recorded after training and consumed by prediction workflows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from density_model.config.config_schema import FeatureConfig, ModelConfig

__all__ = ["TrainingArtifactManifest", "load_training_manifest"]


class TrainingArtifactManifest(BaseModel):
    """
    Frozen manifest of trained artifacts and model schema.

    Parameters
    ----------
    feature_config : FeatureConfig
        Feature engineering and stream configuration used during training.
    model_config : ModelConfig
        Model architecture configuration used during training.
    calendar : str
        Master exchange calendar used by the trained model.
    preprocessing_path : str
        Saved preprocessing bundle path.
    model_state_path : str
        Saved torch state dictionary path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_config: FeatureConfig
    model_config_runtime: ModelConfig
    calendar: str
    preprocessing_path: str
    model_state_path: str

    def preprocessing_artifact_path(self) -> Path:
        """
        Return the saved preprocessing artifact path.

        Returns
        -------
        pathlib.Path
            Preprocessing artifact path.
        """

        return Path(self.preprocessing_path)

    def model_weights_path(self) -> Path:
        """
        Return the saved model weights path.

        Returns
        -------
        pathlib.Path
            Torch model state path.
        """

        return Path(self.model_state_path)

    def save(self, path: str | Path) -> None:
        """
        Save the manifest to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Output JSON path.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def load_training_manifest(path: str | Path) -> TrainingArtifactManifest:
    """
    Load a frozen training manifest from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Manifest JSON path.

    Returns
    -------
    TrainingArtifactManifest
        Parsed training manifest.
    """

    return TrainingArtifactManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
