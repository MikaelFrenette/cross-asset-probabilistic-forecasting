import torch

from dlecosys.shared.data import apply_transforms, fit_transforms


class TestFitTransforms:
    def test_scaler_only(self):
        X = torch.randn(20, 4)
        t = fit_transforms(X, categorical_cols=[], scaling="standard")
        assert t.scaler is not None
        assert t.tokenizers == {}
        assert t.cont_cols == [0, 1, 2, 3]

    def test_tokenizers_only(self):
        X = torch.randint(0, 5, (20, 4)).float()
        t = fit_transforms(X, categorical_cols=[0, 2], scaling=None)
        assert t.scaler is None
        assert set(t.tokenizers.keys()) == {0, 2}
        assert t.cont_cols == [1, 3]

    def test_mixed(self):
        X = torch.cat([torch.randint(0, 5, (20, 1)).float(), torch.randn(20, 3)], dim=1)
        t = fit_transforms(X, categorical_cols=[0], scaling="minmax")
        assert t.scaler is not None
        assert set(t.tokenizers.keys()) == {0}
        assert t.cont_cols == [1, 2, 3]

    def test_no_scaling_no_cat(self):
        X = torch.randn(20, 4)
        t = fit_transforms(X, categorical_cols=[], scaling=None)
        assert t.scaler is None
        assert t.tokenizers == {}


class TestApplyTransforms:
    def test_scaler_changes_values_but_preserves_shape(self):
        torch.manual_seed(0)
        X = torch.randn(50, 4) * 10 + 5
        t = fit_transforms(X, categorical_cols=[], scaling="standard")
        Xt = apply_transforms(X, t)
        assert Xt.shape == X.shape
        # scaled data should have approximately zero mean
        assert abs(Xt.mean().item()) < 0.1

    def test_does_not_mutate_input(self):
        X = torch.randn(20, 4)
        X_copy = X.clone()
        t = fit_transforms(X, categorical_cols=[], scaling="standard")
        _ = apply_transforms(X, t)
        assert torch.equal(X, X_copy)
