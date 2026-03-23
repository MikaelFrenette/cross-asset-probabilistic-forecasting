"""
Configuration Tests
-------------------
Unit tests for typed loading of the public repository workflow configurations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from density_model.config import (
    ExtractConfig,
    GarchBenchmarkConfig,
    PredictConfig,
    TrainConfig,
    load_extract_config,
    load_garch_benchmark_config,
    load_predict_config,
    load_train_config,
)

__all__ = []


def test_load_extract_train_and_predict_configs() -> None:
    """
    Parse the public extraction, training, and prediction YAML files.

    Returns
    -------
    None
        This test asserts the new CLI-facing config surface.
    """

    extract_config = load_extract_config(Path("configs/extract/train_yahoo.yaml"))
    validation_extract_config = load_extract_config(Path("configs/extract/val_yahoo.yaml"))
    garch_config = load_garch_benchmark_config(Path("configs/benchmark/garch.yaml"))
    train_config = load_train_config(Path("configs/train/main.yaml"))
    predict_config = load_predict_config(Path("configs/predict/main.yaml"))

    assert isinstance(extract_config, ExtractConfig)
    assert extract_config.output_path().as_posix() == "data/train_features.csv"
    assert extract_config.data.start_date == "2005-01-01"
    assert isinstance(validation_extract_config, ExtractConfig)
    assert validation_extract_config.output_path().as_posix() == "data/val_features.csv"
    assert validation_extract_config.data.start_date == "2018-01-01"
    assert isinstance(garch_config, GarchBenchmarkConfig)
    assert garch_config.dataset.csv_file().as_posix() == "data/train_features.csv"
    assert garch_config.target_column == "return"
    assert garch_config.spec.mean == "AR"
    assert garch_config.spec.lags == 1
    assert garch_config.artifacts.model_path().as_posix() == "artifacts/demo/garch/garch_benchmark.json"
    assert isinstance(train_config, TrainConfig)
    assert train_config.dataset.train_csv_file().as_posix() == "data/train_features.csv"
    assert train_config.dataset.validation_csv_file().as_posix() == "data/val_features.csv"
    assert train_config.dataset.date_column == "date"
    assert train_config.dataset.id_column == "asset_id"
    assert train_config.features.sequence_length == 60
    assert train_config.features.target_column == "return"
    assert train_config.features.streams.resolve_continuous_columns() == ("return",)
    assert train_config.features.streams.resolve_static_categorical_columns() == ("ticker",)
    assert train_config.model.name == "cross_attention_volatility"
    assert train_config.model.model_dim == 64
    assert train_config.model.ff_dim == 256
    assert train_config.model.likelihood_eps == pytest.approx(1e-6)
    assert train_config.training.learning_rate == pytest.approx(1e-4)
    assert train_config.training.shuffle is True
    assert train_config.training.grad_clip is None
    assert train_config.training.verbose == 2
    assert train_config.early_stopping.enabled is True
    assert train_config.early_stopping.monitor == "val_loss"
    assert train_config.model_checkpoint.enabled is True
    assert train_config.model_checkpoint.monitor == "val_loss"
    assert isinstance(predict_config, PredictConfig)
    assert predict_config.manifest_file().as_posix() == "artifacts/demo/training_manifest.json"
    assert predict_config.dataset.csv_file().as_posix() == "data/predict_features.csv"


def test_load_train_config_rejects_missing_sections(tmp_path: Path) -> None:
    """
    Reject a training YAML payload missing required top-level sections.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test asserts fail-fast training config validation.
    """

    config_path = tmp_path / "broken_train.yaml"
    config_path.write_text("dataset:\n  csv_path: data/train_features.csv\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="Field required"):
        load_train_config(config_path)
