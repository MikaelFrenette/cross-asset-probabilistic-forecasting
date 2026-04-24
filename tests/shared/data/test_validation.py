import pytest
import torch

from dlecosys.shared.data.validation import validate_split, validate_splits


def _ok():
    return torch.randn(10, 4), torch.randn(10, 1)


class TestValidateSplit:
    def test_valid_passes(self):
        validate_split(*_ok(), split="train")

    def test_nan_in_X_raises(self):
        X, y = _ok()
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN.*X"):
            validate_split(X, y, split="train")

    def test_inf_in_X_raises(self):
        X, y = _ok()
        X[0, 0] = float("inf")
        with pytest.raises(ValueError, match="Inf.*X"):
            validate_split(X, y, split="train")

    def test_nan_in_y_raises(self):
        X, y = _ok()
        y[0] = float("nan")
        with pytest.raises(ValueError, match="NaN.*y"):
            validate_split(X, y, split="train")

    def test_inf_in_y_raises(self):
        X, y = _ok()
        y[0] = float("inf")
        with pytest.raises(ValueError, match="Inf.*y"):
            validate_split(X, y, split="val")

    def test_long_y_passes_without_nan_check(self):
        X = torch.randn(10, 4)
        y = torch.randint(0, 3, (10,))
        validate_split(X, y, split="train")  # must not raise

    def test_1d_X_raises(self):
        X = torch.randn(10)
        y = torch.randn(10)
        with pytest.raises(ValueError, match="2D"):
            validate_split(X, y, split="train")

    def test_non_float_X_raises(self):
        X = torch.randint(0, 5, (10, 4))
        y = torch.randn(10, 1)
        with pytest.raises(ValueError, match="float dtype"):
            validate_split(X, y, split="train")

    def test_split_name_in_error_message(self):
        X, y = _ok()
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="my_split"):
            validate_split(X, y, split="my_split")


class TestValidateSplits:
    def test_valid_dict_passes(self):
        splits = {"train": _ok(), "val": _ok(), "test": _ok()}
        validate_splits(splits)

    def test_bad_split_raises(self):
        X, y = _ok()
        X[0, 0] = float("nan")
        splits = {"train": _ok(), "val": (X, y)}
        with pytest.raises(ValueError, match="val"):
            validate_splits(splits)

    def test_tag_appears_in_error(self):
        X, y = _ok()
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="post-transform"):
            validate_splits({"train": (X, y)}, tag="post-transform")
