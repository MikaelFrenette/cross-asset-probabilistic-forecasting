"""
Predict Configuration
---------------------
Typed configuration objects for CSV-based prediction workflows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict
import yaml

from density_model.config.train import CsvDatasetConfig

__all__ = ["PredictConfig", "load_predict_config"]


class PredictConfig(BaseModel):
    """
    Top-level configuration for CSV-based prediction runs.

    Parameters
    ----------
    manifest_path : str
        Frozen training manifest path.
    dataset : CsvDatasetConfig
        Raw extracted feature CSV path and schema information used for prediction.
    output_csv : str
        Prediction output CSV path.
    device : str, default="cpu"
        Torch device used for inference.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: str
    dataset: CsvDatasetConfig
    output_csv: str
    device: str = "cpu"

    def manifest_file(self) -> Path:
        """
        Return the configured training manifest path.

        Returns
        -------
        pathlib.Path
            Manifest path.
        """

        return Path(self.manifest_path)

    def output_path(self) -> Path:
        """
        Return the configured prediction output path.

        Returns
        -------
        pathlib.Path
            Prediction CSV path.
        """

        return Path(self.output_csv)


def load_predict_config(path: str | Path) -> PredictConfig:
    """
    Load a prediction YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    PredictConfig
        Parsed prediction configuration.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(payload, dict):
        raise TypeError("The prediction configuration file must contain a top-level mapping.")
    return PredictConfig.model_validate(payload)
