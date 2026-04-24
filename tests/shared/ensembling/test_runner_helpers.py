import torch

from dlecosys.shared.ensembling.runner import _remap_categorical_cols


class TestRemapCategoricalCols:
    def test_identity_when_all_features_present(self):
        fi = torch.arange(5)
        out = _remap_categorical_cols([0, 2, 4], fi)
        assert out == [0, 2, 4]

    def test_remap_to_subset_positions(self):
        # Full space: cols [0..9]. Subset chose [2, 5, 7, 9] (4 features)
        fi = torch.tensor([2, 5, 7, 9])
        # cat cols in full space are [5, 9]
        out = _remap_categorical_cols([5, 9], fi)
        # In the subset, col 5 is at position 1 and col 9 at position 3
        assert out == [1, 3]

    def test_drops_cat_cols_not_in_subset(self):
        fi = torch.tensor([1, 3, 7])
        # Cat cols [0, 3, 9] — only col 3 is in the subset (position 1)
        out = _remap_categorical_cols([0, 3, 9], fi)
        assert out == [1]

    def test_empty_cat_cols(self):
        fi = torch.arange(5)
        assert _remap_categorical_cols([], fi) == []

    def test_all_cat_cols_dropped(self):
        fi = torch.tensor([1, 2])
        assert _remap_categorical_cols([0, 3, 4], fi) == []
