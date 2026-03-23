"""
GARCH Benchmark
---------------
Artifact-based GARCH benchmark fitting and rolling prediction workflows for
per-asset daily return forecasting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from density_model.config import CsvDatasetConfig, GarchBenchmarkConfig

__all__ = [
    "GarchAssetArtifact",
    "GarchBenchmarkArtifact",
    "fit_garch_benchmark_from_config",
    "predict_garch_from_config",
]


class GarchAssetArtifact(BaseModel):
    """
    Serialized per-asset GARCH benchmark state.

    Parameters
    ----------
    asset_id : str
        Asset identifier.
    parameter_names : tuple of str
        Ordered ARCH parameter names.
    parameters : tuple of float
        Ordered fitted parameter values.
    train_dates : tuple of str
        Historical training dates used to fit the benchmark.
    train_values : tuple of float
        Historical target values used to fit the benchmark.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    train_dates: tuple[str, ...]
    train_values: tuple[float, ...]


class GarchBenchmarkArtifact(BaseModel):
    """
    Serialized fitted GARCH benchmark artifact.

    Parameters
    ----------
    date_column : str
        Public date column name.
    id_column : str
        Public identifier column name.
    target_column : str
        Target column modeled by the benchmark.
    mean : str
        ARCH mean-model identifier.
    vol : str
        ARCH volatility-model identifier.
    p : int
        ARCH autoregressive variance lag order.
    q : int
        ARCH moving-average variance lag order.
    dist : str
        ARCH innovation distribution identifier.
    lags : int or None
        Mean-model lag order used when ``mean="AR"``.
    scale_factor : float
        Internal multiplicative factor applied to returns for numerical stability.
    assets : tuple of GarchAssetArtifact
        Per-asset fitted benchmark states.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    date_column: str
    id_column: str
    target_column: str
    mean: str
    lags: int | None
    vol: str
    p: int
    q: int
    dist: str
    scale_factor: float
    assets: tuple[GarchAssetArtifact, ...]

    def save(self, path: str | Path) -> None:
        """
        Save the benchmark artifact to disk.

        Parameters
        ----------
        path : str or pathlib.Path
            Output JSON path.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> GarchBenchmarkArtifact:
        """
        Load a benchmark artifact from disk.

        Parameters
        ----------
        path : str or pathlib.Path
            JSON artifact path.

        Returns
        -------
        GarchBenchmarkArtifact
            Parsed benchmark artifact.
        """

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def fit_garch_benchmark_from_config(config: GarchBenchmarkConfig) -> Path:
    """
    Fit one GARCH benchmark per asset and save the artifact to disk.

    Parameters
    ----------
    config : GarchBenchmarkConfig
        Typed benchmark configuration.

    Returns
    -------
    pathlib.Path
        Serialized benchmark artifact path.
    """

    arch_model = _load_arch_model()
    frame = _load_dataset(
        csv_path=config.dataset.csv_file(),
        date_column=config.dataset.date_column,
        id_column=config.dataset.id_column,
        target_column=config.target_column,
    )
    scale_factor = 100.0
    assets: list[GarchAssetArtifact] = []
    for asset_id, group in frame.groupby(config.dataset.id_column, sort=True):
        series = group.set_index(config.dataset.date_column)[config.target_column].dropna()
        if len(series) < 5:
            raise ValueError(f"GARCH benchmark fitting requires at least five observations for asset {asset_id!r}.")
        scaled_series = series * scale_factor
        result = arch_model(
            y=scaled_series,
            mean=config.spec.mean,
            lags=config.spec.lags,
            p=config.spec.p,
            q=config.spec.q,
            vol=config.spec.vol,
            dist=config.spec.dist,
        ).fit(disp=False)
        assets.append(
            GarchAssetArtifact(
                asset_id=str(asset_id),
                parameter_names=tuple(map(str, result.params.index.tolist())),
                parameters=tuple(float(value) for value in result.params.to_numpy()),
                train_dates=tuple(pd.Timestamp(index).date().isoformat() for index in series.index),
                train_values=tuple(float(value) for value in series.to_numpy()),
            )
        )

    artifact = GarchBenchmarkArtifact(
        date_column=config.dataset.date_column,
        id_column=config.dataset.id_column,
        target_column=config.target_column,
        mean=config.spec.mean,
        lags=config.spec.lags,
        vol=config.spec.vol,
        p=config.spec.p,
        q=config.spec.q,
        dist=config.spec.dist,
        scale_factor=scale_factor,
        assets=tuple(assets),
    )
    artifact_path = config.artifacts.model_path()
    artifact.save(artifact_path)
    return artifact_path


def predict_garch_from_config(
    *,
    config: GarchBenchmarkConfig,
    dataset_path: str | Path | None = None,
) -> Path:
    """
    Generate rolling GARCH benchmark predictions from a prepared CSV dataset.

    Parameters
    ----------
    config : GarchBenchmarkConfig
        Typed benchmark configuration.
    dataset_path : str or pathlib.Path or None, default=None
        Optional dataset CSV override used for prediction.

    Returns
    -------
    pathlib.Path
        Rolling benchmark prediction CSV path.
    """

    arch_model = _load_arch_model()
    artifact = GarchBenchmarkArtifact.load(config.artifacts.model_path())
    prediction_csv = Path(dataset_path) if dataset_path is not None else config.dataset.csv_file()
    frame = _load_dataset(
        csv_path=prediction_csv,
        date_column=artifact.date_column,
        id_column=artifact.id_column,
        target_column=artifact.target_column,
    )
    rows: list[dict[str, object]] = []
    artifact_by_asset = {asset.asset_id: asset for asset in artifact.assets}
    for asset_id, group in frame.groupby(artifact.id_column, sort=True):
        asset_key = str(asset_id)
        if asset_key not in artifact_by_asset:
            raise ValueError(f"Prediction dataset contains unseen asset {asset_key!r} for the saved GARCH benchmark.")
        asset_artifact = artifact_by_asset[asset_key]
        prediction_series = group.set_index(artifact.date_column)[artifact.target_column].dropna()
        combined_series = pd.Series(
            data=list(asset_artifact.train_values) + prediction_series.tolist(),
            index=pd.to_datetime(list(asset_artifact.train_dates) + [index.date().isoformat() for index in prediction_series.index]),
            dtype=float,
        )
        scaled_combined_series = combined_series * artifact.scale_factor
        model = arch_model(
            y=scaled_combined_series,
            mean=artifact.mean,
            lags=artifact.lags,
            p=artifact.p,
            q=artifact.q,
            vol=artifact.vol,
            dist=artifact.dist,
        )
        fixed_result = model.fix(list(asset_artifact.parameters))
        forecast = fixed_result.forecast(
            horizon=1,
            start=prediction_series.index.min(),
            align="target",
        )
        forecast_mean = forecast.mean.iloc[:, -1].squeeze()
        forecast_sigma = forecast.variance.iloc[:, -1].pow(0.5).squeeze()
        for forecast_date, mu_value in forecast_mean.dropna().items():
            sigma_value = float(forecast_sigma.loc[forecast_date]) / artifact.scale_factor
            rows.append(
                {
                    "forecast_start_date": pd.Timestamp(forecast_date),
                    "forecast_end_date": pd.Timestamp(forecast_date),
                    "asset_id": asset_key,
                    "mu": float(mu_value) / artifact.scale_factor,
                    "sigma": sigma_value,
                }
            )

    output_path = config.artifacts.predictions_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["asset_id", "forecast_start_date"]).to_csv(output_path, index=False)
    return output_path


def _load_arch_model():  # type: ignore[no-untyped-def]
    """
    Import and return the optional ARCH benchmark constructor.

    Returns
    -------
    callable
        Imported ``arch.arch_model`` callable.
    """

    try:
        from arch import arch_model
    except ImportError as error:  # pragma: no cover - optional benchmark dependency.
        raise ImportError("arch must be installed to fit or predict the GARCH benchmark.") from error
    return arch_model


def _load_dataset(
    *,
    csv_path: Path,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Load and validate a benchmark dataset CSV.

    Parameters
    ----------
    csv_path : pathlib.Path
        Dataset CSV path.
    date_column : str
        Date column name.
    id_column : str
        Identifier column name.
    target_column : str
        Target column name.

    Returns
    -------
    pandas.DataFrame
        Validated dataset frame sorted by identifier and date.
    """

    frame = pd.read_csv(csv_path)
    required_columns = {date_column, id_column, target_column}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Benchmark dataset is missing required columns: {missing_text}")
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)
