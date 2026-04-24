"""
Panel Ensemble Manifest
-----------------------
Frozen ensemble artifact manifest recording the surviving estimators,
per-estimator artifact paths, and aggregation policy. Inference reads this
manifest to iterate estimator artifacts and aggregate Gaussian predictions.

Classes
-------
PanelEnsembleEstimatorEntry
    Per-estimator record (index, preprocessing + model paths, manifest path,
    OOB val loss).

PanelEnsembleManifest
    Top-level ensemble manifest serialized to JSON.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PanelEnsembleEstimatorEntry",
    "PanelEnsembleManifest",
    "load_panel_ensemble_manifest",
]


class PanelEnsembleEstimatorEntry(BaseModel):
    """One surviving estimator's artifact pointers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    estimator_manifest_path: str
    oob_val_loss: float | None = None


class PanelEnsembleManifest(BaseModel):
    """Frozen ensemble manifest listing all surviving estimators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ensemble_type: str = "bagging"
    aggregation: str = "mean"
    estimators: list[PanelEnsembleEstimatorEntry]

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


def load_panel_ensemble_manifest(path: str | Path) -> PanelEnsembleManifest:
    """Load and validate a panel ensemble manifest JSON file."""

    return PanelEnsembleManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
