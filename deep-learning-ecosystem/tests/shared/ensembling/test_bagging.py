import pytest
import torch

from dlecosys.shared.config.schema import EnsembleSection, FeatureBootstrapperSection, SampleBootstrapperSection
from dlecosys.shared.ensembling.bagging import BaggingEnsemble
from dlecosys.shared.ensembling.base import BaseEnsemble
from dlecosys.shared.ensembling.bootstrappers import (
    AllFeaturesBootstrapper,
    RandomSubspaceBootstrapper,
    WithReplacementBootstrapper,
)
from dlecosys.shared.ensembling.build import build_ensemble


class TestBaseContract:
    def test_missing_n_estimators_raises(self):
        class BadEnsemble(BaseEnsemble):
            def generate_bootstrap(self, pool_size, estimator_id, rng):
                return None

        with pytest.raises(TypeError, match="n_estimators"):
            BadEnsemble()

    def test_zero_n_estimators_raises(self):
        class ZeroEnsemble(BaseEnsemble):
            n_estimators = 0

            def generate_bootstrap(self, pool_size, estimator_id, rng):
                return None

        with pytest.raises(ValueError, match="positive int"):
            ZeroEnsemble()


class TestBaggingDelegatesToBootstrappers:
    def test_default_bootstrappers(self):
        e = BaggingEnsemble(n_estimators=3)
        assert isinstance(e.sample_bootstrapper, WithReplacementBootstrapper)
        assert isinstance(e.feature_bootstrapper, AllFeaturesBootstrapper)

    def test_injected_bootstrappers_used(self):
        e = BaggingEnsemble(
            n_estimators=3,
            sample_bootstrapper=WithReplacementBootstrapper(max_samples=0.5),
            feature_bootstrapper=RandomSubspaceBootstrapper(max_features=0.7),
        )
        boot, oob = e.generate_bootstrap(100, 0, e.rng_for(0))
        assert boot.numel() == 50  # 0.5 × 100

        fi = e.generate_feature_subset(20, 0, e.rng_for(0))
        assert fi.numel() == 14  # 0.7 × 20

    def test_reproducible_with_same_seed(self):
        e1 = BaggingEnsemble(
            n_estimators=1,
            sample_bootstrapper=WithReplacementBootstrapper(max_samples=1.0),
            seed=7,
        )
        e2 = BaggingEnsemble(
            n_estimators=1,
            sample_bootstrapper=WithReplacementBootstrapper(max_samples=1.0),
            seed=7,
        )
        b1, _ = e1.generate_bootstrap(100, 2, e1.rng_for(2))
        b2, _ = e2.generate_bootstrap(100, 2, e2.rng_for(2))
        assert torch.equal(b1, b2)


class TestBuildEnsemble:
    def test_default_config_builds_bagging(self):
        e = build_ensemble(EnsembleSection(n_estimators=3))
        assert isinstance(e, BaggingEnsemble)
        assert isinstance(e.sample_bootstrapper, WithReplacementBootstrapper)
        assert isinstance(e.feature_bootstrapper, AllFeaturesBootstrapper)

    def test_config_routes_to_random_subspace(self):
        cfg = EnsembleSection(
            n_estimators=3,
            feature_bootstrapper=FeatureBootstrapperSection(type="random_subspace", max_features=0.4),
        )
        e = build_ensemble(cfg)
        assert isinstance(e.feature_bootstrapper, RandomSubspaceBootstrapper)
        assert e.feature_bootstrapper.max_features == 0.4

    def test_config_routes_to_no_bootstrap(self):
        import warnings
        cfg = EnsembleSection(
            n_estimators=3,
            sample_bootstrapper=SampleBootstrapperSection(type="no_bootstrap"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = build_ensemble(cfg)
        from dlecosys.shared.ensembling.bootstrappers import NoBootstrapBootstrapper
        assert isinstance(e.sample_bootstrapper, NoBootstrapBootstrapper)

    def test_invalid_ensemble_type_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EnsembleSection(type="boosting")
