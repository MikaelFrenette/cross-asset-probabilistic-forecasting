"""
Panel Training Manifest
-----------------------
Frozen training-artifact manifest that records everything the inference path
needs to reproduce a trained panel model without re-parsing the training YAML.

Classes
-------
PanelTrainingManifest
    Pydantic-validated JSON manifest emitted at the end of training and
    consumed by the prediction script to rebuild the model and preprocessing
    bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PanelTrainingManifest", "load_panel_manifest"]


class PanelTrainingManifest(BaseModel):
    """Frozen manifest for a completed panel training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    preprocessing_path: str
    model_state_path: str
    model_name: str
    model_params: dict[str, Any]
    sequence_length: int = Field(gt=0)
    forecast_horizon: int = Field(gt=0)
    target_column: str
    date_column: str
    id_column: str
    continuous_columns: tuple[str, ...]
    dynamic_categorical_columns: tuple[str, ...] = ()
    static_categorical_columns: tuple[str, ...] = ()
    calendar: str = "XNYS"

    def preprocessing_artifact_path(self) -> Path:
        return Path(self.preprocessing_path)

    def model_weights_path(self) -> Path:
        return Path(self.model_state_path)

    def save(self, path: str | Path) -> None:
        """Write the manifest to disk as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def load_panel_manifest(path: str | Path) -> PanelTrainingManifest:
    """Load and validate a panel training manifest JSON file."""

    return PanelTrainingManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
