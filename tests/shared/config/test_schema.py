import pytest
from pydantic import ValidationError

from dlecosys.shared.config.schema import (
    DataSection,
    ExperimentSection,
    ModelSection,
    OptimizerSection,
    PipelineConfig,
    TrainingSection,
)


class TestExperimentSection:
    def test_required_fields(self):
        s = ExperimentSection(name="run_1")
        assert s.seed == 42
        assert s.output_dir == "outputs"

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            ExperimentSection(seed=123)


class TestDataSection:
    def test_regression_ok(self):
        d = DataSection(task="regression", path="data/my_data.csv")
        assert d.target_col == "target"
        assert d.scaling == "standard"

    def test_classification_ok(self):
        d = DataSection(task="classification", path="data/my_data.csv")
        assert d.task == "classification"

    def test_invalid_task_raises(self):
        with pytest.raises(ValidationError):
            DataSection(task="forecasting", path="data/my_data.csv")

    def test_missing_path_raises(self):
        with pytest.raises(ValidationError):
            DataSection(task="regression")

    def test_invalid_scaling_raises(self):
        with pytest.raises(ValidationError):
            DataSection(task="regression", path="data/x.csv", scaling="zscore")


class TestOptimizerSection:
    def test_defaults(self):
        o = OptimizerSection()
        assert o.name == "adam"
        assert o.lr == 1e-3

    def test_invalid_name_raises(self):
        with pytest.raises(ValidationError):
            OptimizerSection(name="rmsprop")


class TestTrainingSection:
    def test_defaults(self):
        t = TrainingSection()
        assert t.epochs == 50
        assert t.amp is False
        assert t.grad_accumulation_steps == 1

    def test_invalid_loss_raises(self):
        with pytest.raises(ValidationError):
            TrainingSection(loss="huber")

    def test_invalid_grad_accum_raises(self):
        with pytest.raises(ValidationError):
            TrainingSection(grad_accumulation_steps=0)


class TestPipelineConfig:
    def _base_kwargs(self):
        return {
            "experiment": {"name": "run_1"},
            "data": {"task": "regression", "path": "data/x.csv"},
            "model": {"name": "mlp", "params": {"input_dim": 4}},
            "training": {},
        }

    def test_valid_regression_config(self):
        cfg = PipelineConfig(**self._base_kwargs())
        assert cfg.experiment.name == "run_1"

    def test_valid_classification_config(self):
        kw = self._base_kwargs()
        kw["data"] = {"task": "classification", "path": "data/x.csv"}
        cfg = PipelineConfig(**kw)
        assert cfg.data.task == "classification"

    def test_inference_defaults_applied(self):
        cfg = PipelineConfig(**self._base_kwargs())
        assert cfg.inference.batch_size == 64
