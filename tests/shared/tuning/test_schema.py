import pytest
from pydantic import ValidationError

from dlecosys.shared.config.schema import PrunerSection, TuningSection


class TestPrunerSection:
    def test_defaults(self):
        p = PrunerSection()
        assert p.enabled is False
        assert p.type == "median"
        assert p.n_warmup_steps == 5
        assert p.n_startup_trials == 5

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            PrunerSection(type="hyperband")


class TestTuningSection:
    def _base(self, **overrides):
        data = {
            "study_name": "s",
            "search_space": {"training.optimizer.lr": [1e-4, 1e-3]},
        }
        data.update(overrides)
        return data

    def test_defaults(self):
        t = TuningSection(**self._base())
        assert t.direction == "minimize"
        assert t.sampler == "grid"
        assert t.metric == "val_loss"
        assert t.n_trials is None
        assert t.storage is None

    def test_invalid_sampler_raises(self):
        with pytest.raises(ValidationError):
            TuningSection(**self._base(sampler="tpe"))

    def test_invalid_direction_raises(self):
        with pytest.raises(ValidationError):
            TuningSection(**self._base(direction="down"))

    def test_empty_search_space_raises(self):
        with pytest.raises(ValidationError, match="at least one entry"):
            TuningSection(study_name="s", search_space={})

    def test_empty_choices_raises(self):
        with pytest.raises(ValidationError, match="no choices"):
            TuningSection(study_name="s", search_space={"x": []})

    def test_list_choice_values(self):
        t = TuningSection(
            study_name="s",
            search_space={"model.params.hidden_dims": [[64, 32], [128, 64]]},
        )
        assert t.search_space["model.params.hidden_dims"] == [[64, 32], [128, 64]]
