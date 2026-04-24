import pytest
import yaml
from pathlib import Path

from dlecosys.shared.config.loader import ConfigError, load_config


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _base_data():
    return {
        "base": None,
        "experiment": {"name": "r1"},
        "data": {"task": "regression", "path": "data/x.csv"},
        "model": {"name": "mlp", "params": {"input_dim": 4}},
        "training": {},
    }


class TestLoadConfigStandalone:
    def test_loads_valid_config(self, tmp_path):
        p = _write(tmp_path / "cfg.yaml", _base_data())
        cfg = load_config(p)
        assert cfg.experiment.name == "r1"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_invalid_schema_raises(self, tmp_path):
        data = _base_data()
        data["training"] = {"loss": "unknown_loss"}
        p = _write(tmp_path / "bad.yaml", data)
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_config(p)


class TestLoadConfigInheritance:
    def test_base_fields_inherited(self, tmp_path):
        base = _base_data()
        p_base = _write(tmp_path / "base.yaml", base)

        override = {
            "base": "base.yaml",
            "experiment": {"name": "r2"},
        }
        p_child = _write(tmp_path / "child.yaml", override)

        cfg = load_config(p_child)
        assert cfg.experiment.name == "r2"
        assert cfg.data.task == "regression"

    def test_override_wins_over_base(self, tmp_path):
        base = _base_data()
        base["training"] = {"epochs": 10}
        p_base = _write(tmp_path / "base.yaml", base)

        override = {
            "base": "base.yaml",
            "experiment": {"name": "rx"},
            "data": {"task": "regression"},
            "model": {"name": "mlp", "params": {"input_dim": 4}},
            "training": {"epochs": 99},
        }
        p_child = _write(tmp_path / "child.yaml", override)

        cfg = load_config(p_child)
        assert cfg.training.epochs == 99

    def test_deep_merge_preserves_unoverridden_keys(self, tmp_path):
        base = _base_data()
        base["training"] = {"epochs": 5, "amp": True}
        p_base = _write(tmp_path / "base.yaml", base)

        override = {
            "base": "base.yaml",
            "experiment": {"name": "rx"},
            "data": {"task": "regression"},
            "model": {"name": "mlp", "params": {"input_dim": 4}},
            "training": {"epochs": 20},
        }
        p_child = _write(tmp_path / "child.yaml", override)

        cfg = load_config(p_child)
        assert cfg.training.epochs == 20
        assert cfg.training.amp is True

    def test_two_level_inheritance_raises(self, tmp_path):
        grandparent = _base_data()
        p_gp = _write(tmp_path / "grandparent.yaml", grandparent)

        parent = {"base": "grandparent.yaml", "experiment": {"name": "rp"}}
        p_parent = _write(tmp_path / "parent.yaml", parent)

        child = {"base": "parent.yaml", "experiment": {"name": "rc"}}
        p_child = _write(tmp_path / "child.yaml", child)

        with pytest.raises(ConfigError, match="1 level"):
            load_config(p_child)

    def test_missing_base_file_raises(self, tmp_path):
        override = {
            "base": "does_not_exist.yaml",
            "experiment": {"name": "rx"},
        }
        p_child = _write(tmp_path / "child.yaml", override)
        with pytest.raises(FileNotFoundError):
            load_config(p_child)
