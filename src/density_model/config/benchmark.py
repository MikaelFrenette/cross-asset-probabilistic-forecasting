"""
Benchmark Configuration
-----------------------
Typed configuration objects for GARCH benchmark fitting and prediction
workflows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
import yaml

from density_model.config.train import CsvDatasetConfig

__all__ = ["GarchArtifactConfig", "GarchBenchmarkConfig", "GarchSpecConfig", "load_garch_benchmark_config"]


class GarchSpecConfig(BaseModel):
    """
    GARCH model specification used by the benchmark workflow.

    Parameters
    ----------
    mean : {"zero", "constant", "AR"}, default="zero"
        Mean model passed to ``arch.arch_model``.
    lags : int or None, default=None
        Mean-model lag order used when ``mean="AR"``.
    vol : {"GARCH"}, default="GARCH"
        Volatility model identifier.
    p : int, default=1
        Autoregressive lag order of the variance equation.
    q : int, default=1
        Moving-average lag order of the variance equation.
    dist : {"normal", "t"}, default="normal"
        Innovation distribution identifier.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    mean: str = Field(default="zero", pattern="^(zero|constant|AR|ar)$")
    lags: int | None = Field(default=None, ge=1)
    vol: str = Field(default="GARCH", pattern="^(GARCH)$")
    p: int = Field(default=1, ge=1)
    q: int = Field(default=1, ge=1)
    dist: str = Field(default="normal", pattern="^(normal|t)$")

    @field_validator("mean")
    @classmethod
    def normalize_mean_name(cls, value: str) -> str:
        """
        Normalize supported ARCH mean model identifiers.

        Parameters
        ----------
        value : str
            Candidate mean-model identifier.

        Returns
        -------
        str
            Normalized mean-model identifier.
        """

        return "AR" if value.upper() == "AR" else value.lower()

    def model_post_init(self, __context: object) -> None:
        """
        Validate lag usage for the selected mean model.

        Parameters
        ----------
        __context : object
            Pydantic post-init context.
        """

        if self.mean == "AR" and self.lags is None:
            raise ValueError("spec.lags must be provided when spec.mean='AR'.")
        if self.mean != "AR" and self.lags is not None:
            raise ValueError("spec.lags is only supported when spec.mean='AR'.")


class GarchArtifactConfig(BaseModel):
    """
    Artifact configuration for the GARCH benchmark workflow.

    Parameters
    ----------
    output_dir : str
        Directory where benchmark artifacts are written.
    model_filename : str, default="garch_benchmark.json"
        Serialized fitted benchmark artifact filename.
    predictions_filename : str, default="garch_predictions.csv"
        Default rolling prediction CSV filename.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: str
    model_filename: str = "garch_benchmark.json"
    predictions_filename: str = "garch_predictions.csv"

    @field_validator("output_dir", "model_filename", "predictions_filename")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """
        Validate required benchmark artifact text fields.

        Parameters
        ----------
        value : str
            Candidate string value.

        Returns
        -------
        str
            Validated string value.
        """

        if not value or value.strip() != value:
            raise ValueError("Benchmark artifact fields must be non-empty and must not include surrounding whitespace.")
        return value

    def output_path(self) -> Path:
        """
        Return the configured artifact directory path.

        Returns
        -------
        pathlib.Path
            Artifact directory path.
        """

        return Path(self.output_dir)

    def model_path(self) -> Path:
        """
        Return the configured benchmark artifact path.

        Returns
        -------
        pathlib.Path
            Serialized benchmark artifact path.
        """

        return self.output_path() / self.model_filename

    def predictions_path(self) -> Path:
        """
        Return the configured benchmark prediction CSV path.

        Returns
        -------
        pathlib.Path
            Rolling benchmark prediction CSV path.
        """

        return self.output_path() / self.predictions_filename


class GarchBenchmarkConfig(BaseModel):
    """
    Top-level configuration for GARCH benchmark workflows.

    Parameters
    ----------
    dataset : CsvDatasetConfig
        Training dataset schema and CSV path used to fit the benchmark.
    target_column : str, default="return"
        Supervised target column modeled by the benchmark.
    spec : GarchSpecConfig
        GARCH model specification.
    artifacts : GarchArtifactConfig
        Artifact output configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: CsvDatasetConfig
    target_column: str = "return"
    spec: GarchSpecConfig = Field(default_factory=GarchSpecConfig)
    artifacts: GarchArtifactConfig

    @field_validator("target_column")
    @classmethod
    def validate_target_column(cls, value: str) -> str:
        """
        Validate the configured benchmark target column.

        Parameters
        ----------
        value : str
            Candidate target column name.

        Returns
        -------
        str
            Validated target column name.
        """

        if not value or value.strip() != value:
            raise ValueError("target_column must be a non-empty string without surrounding whitespace.")
        return value


def load_garch_benchmark_config(path: str | Path) -> GarchBenchmarkConfig:
    """
    Load a GARCH benchmark YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    GarchBenchmarkConfig
        Parsed benchmark configuration.
    """

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Configuration file is empty.")
    if not isinstance(payload, dict):
        raise TypeError("The GARCH benchmark configuration file must contain a top-level mapping.")
    return GarchBenchmarkConfig.model_validate(payload)
