import pytest
from pydantic import ValidationError

from dlecosys.shared.config.schema import EnsembleDistributedSection, EnsembleSection


class TestEnsembleSection:
    def test_defaults(self):
        e = EnsembleSection()
        assert e.type == "bagging"
        assert e.n_estimators == 20
        assert e.aggregation == "mean"
        assert e.seed == 42
        assert e.sample_bootstrapper.type == "with_replacement"
        assert e.sample_bootstrapper.max_samples == 1.0
        assert e.feature_bootstrapper.type == "all"
        assert e.feature_bootstrapper.max_features == 1.0
        assert e.distributed.enabled is False
        assert e.distributed.backend == "nccl"

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError):
            EnsembleSection(type="boosting")

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValidationError):
            EnsembleSection(aggregation="geometric")

    def test_invalid_sample_bootstrapper_type(self):
        from dlecosys.shared.config.schema import SampleBootstrapperSection
        with pytest.raises(ValidationError):
            SampleBootstrapperSection(type="knn")

    def test_invalid_feature_bootstrapper_type(self):
        from dlecosys.shared.config.schema import FeatureBootstrapperSection
        with pytest.raises(ValidationError):
            FeatureBootstrapperSection(type="pca")

    def test_override_n_estimators(self):
        e = EnsembleSection(n_estimators=50)
        assert e.n_estimators == 50


class TestEnsembleDistributedSection:
    def test_defaults(self):
        d = EnsembleDistributedSection()
        assert d.enabled is False
        assert d.backend == "nccl"

    def test_gloo(self):
        d = EnsembleDistributedSection(backend="gloo")
        assert d.backend == "gloo"
