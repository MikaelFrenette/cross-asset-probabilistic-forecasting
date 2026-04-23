import pytest

from dlecosys.shared.config.schema import EnsemblePruningSection
from dlecosys.shared.ensembling.pruning import select_estimators


def _results(values):
    """Build a list of per-estimator dicts from a list of OOB values (or None)."""
    return [
        {
            "estimator_id": i,
            "oob_value": v,
            "bootstrap_size": 100,
            "oob_size": 40,
            "feature_count": 10,
        }
        for i, v in enumerate(values)
    ]


class TestPruningDisabled:
    def test_keeps_all_valid_estimators(self):
        cfg = EnsemblePruningSection(enabled=False)
        res = select_estimators(cfg, _results([0.5, 0.3, 0.8]))
        assert res.strategy == "none"
        assert set(res.selected) == {0, 1, 2}

    def test_drops_none_values_even_when_disabled(self):
        cfg = EnsemblePruningSection(enabled=False)
        res = select_estimators(cfg, _results([0.5, None, 0.8]))
        # estimator 1 has no OOB value; it's excluded even without pruning
        assert set(res.selected) == {0, 2}

    def test_n_total_reflects_raw_count(self):
        cfg = EnsemblePruningSection(enabled=False)
        res = select_estimators(cfg, _results([0.5, 0.3]))
        assert res.n_total == 2


class TestTopNMinimize:
    def test_keeps_top_n_by_value(self):
        cfg = EnsemblePruningSection(enabled=True, keep=2, strategy="top_n", direction="minimize")
        res = select_estimators(cfg, _results([0.5, 0.1, 0.8, 0.2]))
        # Ascending sort → 0.1 (id=1), 0.2 (id=3), 0.5 (id=0), 0.8 (id=2)
        assert res.selected == [1, 3]

    def test_keep_exceeds_available_returns_all(self):
        cfg = EnsemblePruningSection(enabled=True, keep=100, strategy="top_n")
        res = select_estimators(cfg, _results([0.5, 0.1, 0.8]))
        assert len(res.selected) == 3

    def test_excludes_none_values(self):
        cfg = EnsemblePruningSection(enabled=True, keep=3, strategy="top_n")
        res = select_estimators(cfg, _results([0.5, None, 0.3, None, 0.1]))
        # Valid: {0, 2, 4} ranked → [4 (0.1), 2 (0.3), 0 (0.5)]
        assert res.selected == [4, 2, 0]


class TestTopNMaximize:
    def test_inverts_ordering(self):
        cfg = EnsemblePruningSection(enabled=True, keep=2, strategy="top_n", direction="maximize")
        res = select_estimators(cfg, _results([0.5, 0.1, 0.8, 0.2]))
        # Descending sort → 0.8 (id=2), 0.5 (id=0)
        assert res.selected == [2, 0]


class TestInvalidStrategy:
    def test_unknown_strategy_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            EnsemblePruningSection(strategy="greedy")
