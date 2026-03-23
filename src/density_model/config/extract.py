"""
Extraction Configuration
------------------------
Typed configuration objects for Yahoo feature extraction workflows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
import yaml

from density_model.config.config_schema import DataConfig, FeatureStreamConfig

__all__ = ["ExtractConfig", "load_extract_config"]


class ExtractConfig(BaseModel):
    """
    Top-level configuration for Yahoo feature extraction.

    Parameters
    ----------
    data : DataConfig
        Yahoo Finance download configuration.
    streams : FeatureStreamConfig
        Public feature-stream configuration describing extracted columns.
    output_csv : str
        Output CSV path used to save extracted raw features.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: DataConfig
    streams: FeatureStreamConfig = Field(
        default_factory=lambda: FeatureStreamConfig(
            continuous=("returns",),
            dynamic_categoricals=None,
            static_categoricals=("ticker",),
        )
    )
    output_csv: str

    def output_path(self) -> Path:
        """
        Return the configured feature extraction output path.

        Returns
        -------
        pathlib.Path
            Output CSV path.
        """

        return Path(self.output_csv)


def load_extract_config(path: str | Path) -> ExtractConfig:
    """
    Load a Yahoo feature extraction YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    ExtractConfig
        Parsed extraction configuration.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(payload, dict):
        raise TypeError("The extraction configuration file must contain a top-level mapping.")
    return ExtractConfig.model_validate(payload)
