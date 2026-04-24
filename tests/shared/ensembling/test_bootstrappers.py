import warnings

import pytest
import torch

from dlecosys.shared.ensembling.bootstrappers import (
    AllFeaturesBootstrapper,
    BaseFeatureBootstrapper,
    BaseSampleBootstrapper,
    NoBootstrapBootstrapper,
    RandomSubspaceBootstrapper,
    WithReplacementBootstrapper,
    build_feature_bootstrapper,
    build_sample_bootstrapper,
)


# ---------------------------------------------------------------------------
# Sample bootstrappers
# ---------------------------------------------------------------------------


class TestWithReplacement:
    def _rng(self):
        return torch.Generator().manual_seed(42)

    def test_bootstrap_size_matches_max_samples(self):
        b = WithReplacementBootstrapper(max_samples=0.5)
        boot, _ = b.sample(200, 0, self._rng())
        assert boot.numel() == 100

    def test_oob_is_complement(self):
        b = WithReplacementBootstrapper(max_samples=1.0)
        boot, oob = b.sample(200, 0, self._rng())
        in_bag = set(boot.tolist())
        oob_set = set(oob.tolist())
        assert in_bag.isdisjoint(oob_set)
        assert in_bag | oob_set == set(range(200))

    def test_expected_oob_fraction(self):
        b = WithReplacementBootstrapper(max_samples=1.0)
        _, oob = b.sample(10000, 0, self._rng())
        assert 0.33 < oob.numel() / 10000 < 0.40

    def test_zero_max_samples_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            WithReplacementBootstrapper(max_samples=0.0)

    def test_empty_bootstrap_raises(self):
        b = WithReplacementBootstrapper(max_samples=0.001)
        with pytest.raises(ValueError, match="must be ≥ 1"):
            b.sample(10, 0, self._rng())

    def test_empty_oob_raises(self):
        b = WithReplacementBootstrapper(max_samples=1.0)
        with pytest.raises(ValueError, match="no OOB"):
            b.sample(1, 0, self._rng())

    def test_isinstance_base(self):
        assert isinstance(WithReplacementBootstrapper(), BaseSampleBootstrapper)


class TestNoBootstrap:
    def test_warns_at_construction(self):
        with pytest.warns(UserWarning, match="no OOB"):
            NoBootstrapBootstrapper()

    def test_returns_full_pool_and_empty_oob(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = NoBootstrapBootstrapper()
        boot, oob = b.sample(50, 0, torch.Generator().manual_seed(0))
        assert torch.equal(boot, torch.arange(50))
        assert oob.numel() == 0
        assert oob.dtype == torch.long


# ---------------------------------------------------------------------------
# Feature bootstrappers
# ---------------------------------------------------------------------------


class TestAllFeatures:
    def test_returns_natural_order(self):
        b = AllFeaturesBootstrapper()
        fi = b.sample(10, 0, torch.Generator().manual_seed(0))
        assert torch.equal(fi, torch.arange(10))

    def test_isinstance_base(self):
        assert isinstance(AllFeaturesBootstrapper(), BaseFeatureBootstrapper)


class TestRandomSubspace:
    def _rng(self):
        return torch.Generator().manual_seed(42)

    def test_subset_size(self):
        b = RandomSubspaceBootstrapper(max_features=0.5)
        fi = b.sample(10, 0, self._rng())
        assert fi.numel() == 5

    def test_returns_sorted_unique(self):
        b = RandomSubspaceBootstrapper(max_features=0.6)
        fi = b.sample(20, 0, self._rng())
        assert torch.equal(fi, fi.sort().values)
        assert len(set(fi.tolist())) == fi.numel()

    def test_full_permutes_all(self):
        b = RandomSubspaceBootstrapper(max_features=1.0)
        fi = b.sample(8, 0, self._rng())
        assert set(fi.tolist()) == set(range(8))

    def test_different_estimators_different_subsets(self):
        b = RandomSubspaceBootstrapper(max_features=0.6)
        a = tuple(b.sample(20, 0, torch.Generator().manual_seed(42)).tolist())
        c = tuple(b.sample(20, 1, torch.Generator().manual_seed(43)).tolist())
        assert a != c

    def test_invalid_max_features_raises(self):
        with pytest.raises(ValueError, match="in .0, 1."):
            RandomSubspaceBootstrapper(max_features=1.5)

    def test_zero_max_features_raises(self):
        with pytest.raises(ValueError):
            RandomSubspaceBootstrapper(max_features=0.0)


# ---------------------------------------------------------------------------
# Build dispatch
# ---------------------------------------------------------------------------


class TestBuildDispatch:
    def test_build_with_replacement(self):
        from dlecosys.shared.config.schema import SampleBootstrapperSection
        b = build_sample_bootstrapper(
            SampleBootstrapperSection(type="with_replacement", max_samples=0.8)
        )
        assert isinstance(b, WithReplacementBootstrapper)
        assert b.max_samples == 0.8

    def test_build_no_bootstrap(self):
        from dlecosys.shared.config.schema import SampleBootstrapperSection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = build_sample_bootstrapper(SampleBootstrapperSection(type="no_bootstrap"))
        assert isinstance(b, NoBootstrapBootstrapper)

    def test_build_all_features(self):
        from dlecosys.shared.config.schema import FeatureBootstrapperSection
        b = build_feature_bootstrapper(FeatureBootstrapperSection(type="all"))
        assert isinstance(b, AllFeaturesBootstrapper)

    def test_build_random_subspace(self):
        from dlecosys.shared.config.schema import FeatureBootstrapperSection
        b = build_feature_bootstrapper(
            FeatureBootstrapperSection(type="random_subspace", max_features=0.6)
        )
        assert isinstance(b, RandomSubspaceBootstrapper)
        assert b.max_features == 0.6


class TestCustomSubclass:
    def test_user_sample_bootstrapper(self):
        """Prove the extension contract works for a custom sampling scheme."""

        class EvenOnlyBootstrapper(BaseSampleBootstrapper):
            def sample(self, pool_size, estimator_id, rng):
                boot = torch.arange(0, pool_size, 2)
                oob = torch.arange(1, pool_size, 2)
                return boot, oob

        b = EvenOnlyBootstrapper()
        boot, oob = b.sample(10, 0, torch.Generator().manual_seed(0))
        assert torch.equal(boot, torch.tensor([0, 2, 4, 6, 8]))
        assert torch.equal(oob, torch.tensor([1, 3, 5, 7, 9]))

    def test_user_feature_bootstrapper(self):
        class FirstHalfBootstrapper(BaseFeatureBootstrapper):
            def sample(self, n_features, estimator_id, rng):
                return torch.arange(n_features // 2)

        b = FirstHalfBootstrapper()
        fi = b.sample(10, 0, torch.Generator().manual_seed(0))
        assert torch.equal(fi, torch.arange(5))
