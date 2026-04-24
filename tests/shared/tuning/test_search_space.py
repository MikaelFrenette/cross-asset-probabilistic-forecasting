import pytest

from dlecosys.shared.tuning.search_space import (
    apply_suggestion,
    from_hashable,
    to_hashable,
)


class TestApplySuggestion:
    def test_sets_leaf_value(self):
        cfg = {"training": {"optimizer": {"lr": 0.001}}}
        apply_suggestion(cfg, "training.optimizer.lr", 0.01)
        assert cfg["training"]["optimizer"]["lr"] == 0.01

    def test_sets_top_level(self):
        cfg = {"epochs": 10}
        apply_suggestion(cfg, "epochs", 50)
        assert cfg["epochs"] == 50

    def test_accepts_list_value(self):
        cfg = {"model": {"params": {"hidden_dims": [32]}}}
        apply_suggestion(cfg, "model.params.hidden_dims", [64, 32])
        assert cfg["model"]["params"]["hidden_dims"] == [64, 32]

    def test_missing_path_raises(self):
        cfg = {"training": {}}
        with pytest.raises(KeyError, match="optimizer"):
            apply_suggestion(cfg, "training.optimizer.lr", 0.01)


class TestHashableRoundtrip:
    def test_list_to_tuple(self):
        assert to_hashable([1, 2, 3]) == (1, 2, 3)

    def test_nested_list_to_tuple(self):
        assert to_hashable([[64, 32], [128, 64]]) == ((64, 32), (128, 64))

    def test_scalar_unchanged(self):
        assert to_hashable(0.01) == 0.01
        assert to_hashable("adam") == "adam"

    def test_tuple_to_list(self):
        assert from_hashable((1, 2, 3)) == [1, 2, 3]

    def test_nested_tuple_to_list(self):
        assert from_hashable(((64, 32), (128, 64))) == [[64, 32], [128, 64]]

    def test_roundtrip(self):
        original = [[64, 32], [128, 64]]
        assert from_hashable(to_hashable(original)) == original
