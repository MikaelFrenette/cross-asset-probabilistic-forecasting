import pytest
from pydantic import ValidationError

from dlecosys.shared.config.schema import DistributedSection, PipelineConfig


class TestDistributedSection:
    def test_defaults(self):
        cfg = DistributedSection()
        assert cfg.enabled is False
        assert cfg.backend == "nccl"

    def test_gloo_backend(self):
        cfg = DistributedSection(backend="gloo")
        assert cfg.backend == "gloo"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValidationError):
            DistributedSection(backend="mpi")

    def test_enabled_true(self):
        cfg = DistributedSection(enabled=True)
        assert cfg.enabled is True


class TestPipelineConfigDistributed:
    def test_distributed_defaults_in_pipeline(self, tmp_path):
        import yaml
        data = {
            "experiment": {"name": "r1"},
            "data": {"task": "regression", "path": "data/x.csv"},
            "model": {"name": "mlp", "params": {"input_dim": 4}},
            "training": {},
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(data))

        from dlecosys.shared.config.loader import load_config
        cfg = load_config(p)
        assert cfg.distributed.enabled is False
        assert cfg.distributed.backend == "nccl"
