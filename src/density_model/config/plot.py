"""
Plot Configuration
------------------
Typed configuration objects for plotting rolling forecast outputs against
realized targets and GARCH reference series.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict
import yaml

from density_model.config.train import CsvDatasetConfig

__all__ = ["PlotConfig", "build_plot_config", "load_plot_config"]


class PlotConfig(BaseModel):
    """
    Top-level configuration for forecast plotting workflows.

    Parameters
    ----------
    manifest_path : str
        Frozen training manifest path.
    predictions_csv : str
        CSV path containing rolling forecast outputs.
    benchmark_predictions_csv : str or None, default=None
        Optional CSV path containing rolling benchmark forecast outputs.
    dataset : CsvDatasetConfig
        Realized dataset CSV path and schema information.
    output_dir : str
        Directory where per-asset plots will be written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_path: str
    predictions_csv: str
    benchmark_predictions_csv: str | None = None
    dataset: CsvDatasetConfig
    output_dir: str

    def manifest_file(self) -> Path:
        """
        Return the configured training manifest path.

        Returns
        -------
        pathlib.Path
            Manifest path.
        """

        return Path(self.manifest_path)

    def predictions_file(self) -> Path:
        """
        Return the configured rolling prediction CSV path.

        Returns
        -------
        pathlib.Path
            Prediction CSV path.
        """

        return Path(self.predictions_csv)

    def output_path(self) -> Path:
        """
        Return the configured plot output directory.

        Returns
        -------
        pathlib.Path
            Plot output directory.
        """

        return Path(self.output_dir)

    def benchmark_predictions_file(self) -> Path | None:
        """
        Return the configured benchmark prediction CSV path when provided.

        Returns
        -------
        pathlib.Path or None
            Optional benchmark prediction CSV path.
        """

        if self.benchmark_predictions_csv is None:
            return None
        return Path(self.benchmark_predictions_csv)


def build_plot_config(
    *,
    manifest_path: str,
    predictions_csv: str,
    benchmark_predictions_csv: str | None,
    dataset_csv: str,
    output_dir: str,
    date_column: str = "date",
    id_column: str = "asset_id",
) -> PlotConfig:
    """
    Build a plotting configuration directly from CLI-style arguments.

    Parameters
    ----------
    manifest_path : str
        Frozen training manifest path.
    predictions_csv : str
        Model prediction CSV path.
    benchmark_predictions_csv : str or None
        Optional benchmark prediction CSV path.
    dataset_csv : str
        Realized dataset CSV path.
    output_dir : str
        Plot output directory.
    date_column : str, default="date"
        Realized dataset date column name.
    id_column : str, default="asset_id"
        Realized dataset identifier column name.

    Returns
    -------
    PlotConfig
        Parsed plotting configuration object.
    """

    return PlotConfig(
        manifest_path=manifest_path,
        predictions_csv=predictions_csv,
        benchmark_predictions_csv=benchmark_predictions_csv,
        dataset=CsvDatasetConfig(
            csv_path=dataset_csv,
            date_column=date_column,
            id_column=id_column,
        ),
        output_dir=output_dir,
    )


def load_plot_config(path: str | Path) -> PlotConfig:
    """
    Load a plotting YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    PlotConfig
        Parsed plotting configuration.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(payload, dict):
        raise TypeError("The plotting configuration file must contain a top-level mapping.")
    return PlotConfig.model_validate(payload)
