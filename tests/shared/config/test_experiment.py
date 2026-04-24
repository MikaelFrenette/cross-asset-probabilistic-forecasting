from dlecosys.shared.config.schema import ExperimentSection


class TestExperimentDefaults:
    def test_deterministic_default_true(self):
        s = ExperimentSection(name="x")
        assert s.deterministic is True

    def test_deterministic_override(self):
        s = ExperimentSection(name="x", deterministic=False)
        assert s.deterministic is False
