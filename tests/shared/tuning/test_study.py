import pytest

optuna = pytest.importorskip("optuna")

from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import GridSampler, RandomSampler

from dlecosys.shared.config.schema import PrunerSection, TuningSection
from dlecosys.shared.tuning.study import build_pruner, build_sampler, build_study


def _tuning(**kwargs) -> TuningSection:
    data = {
        "study_name": "test_study",
        "search_space": {
            "training.optimizer.lr": [1e-4, 1e-3],
            "model.params.dropout": [0.0, 0.2],
        },
    }
    data.update(kwargs)
    return TuningSection(**data)


class TestBuildSampler:
    def test_grid_sampler(self):
        s = build_sampler(_tuning(sampler="grid"))
        assert isinstance(s, GridSampler)

    def test_random_sampler(self):
        s = build_sampler(_tuning(sampler="random"))
        assert isinstance(s, RandomSampler)


class TestBuildPruner:
    def test_disabled_returns_nop(self):
        p = build_pruner(_tuning(pruner=PrunerSection(enabled=False)))
        assert isinstance(p, NopPruner)

    def test_median_enabled(self):
        p = build_pruner(_tuning(pruner=PrunerSection(enabled=True, type="median")))
        assert isinstance(p, MedianPruner)

    def test_type_none_returns_nop_even_if_enabled(self):
        p = build_pruner(_tuning(pruner=PrunerSection(enabled=True, type="none")))
        assert isinstance(p, NopPruner)


class TestBuildStudy:
    def test_creates_study_with_direction(self):
        study = build_study(_tuning(direction="minimize"))
        assert study.direction.name == "MINIMIZE"

    def test_maximize_direction(self):
        study = build_study(_tuning(direction="maximize"))
        assert study.direction.name == "MAXIMIZE"
