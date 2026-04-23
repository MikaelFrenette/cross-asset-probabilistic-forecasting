import torch
from torch.utils.data import TensorDataset

from dlecosys.shared.config.schema import DataSection
from dlecosys.shared.data import preprocess_fold


def _data_cfg(**kw):
    base = dict(task="regression", path="x.csv", scaling="standard", categorical_cols=[])
    base.update(kw)
    return DataSection(**base)


class TestPreprocessFold:
    def test_scales_using_train_stats_only(self):
        torch.manual_seed(0)
        X_train = torch.randn(50, 4) * 3 + 10
        y_train = torch.randn(50, 1)
        X_val = torch.randn(10, 4) * 3 + 10
        y_val = torch.randn(10, 1)

        train_ds = TensorDataset(X_train, y_train)
        val_ds = TensorDataset(X_val, y_val)

        train_out, val_out = preprocess_fold(_data_cfg(), train_ds, val_ds)

        # train ds mean should be ~0; val mean inherits train stats (not necessarily 0)
        Xt, _ = train_out.tensors
        assert abs(Xt.mean().item()) < 0.1

    def test_preserves_shapes(self):
        X_train = torch.randn(30, 5)
        y_train = torch.randn(30, 1)
        X_val = torch.randn(7, 5)
        y_val = torch.randn(7, 1)

        train_out, val_out = preprocess_fold(
            _data_cfg(),
            TensorDataset(X_train, y_train),
            TensorDataset(X_val, y_val),
        )
        assert train_out.tensors[0].shape == (30, 5)
        assert val_out.tensors[0].shape == (7, 5)

    def test_null_scaling_passthrough(self):
        X_train = torch.randn(30, 4)
        X_val = torch.randn(7, 4)
        y_train = torch.randn(30, 1)
        y_val = torch.randn(7, 1)

        train_out, val_out = preprocess_fold(
            _data_cfg(scaling=None),
            TensorDataset(X_train, y_train),
            TensorDataset(X_val, y_val),
        )
        # with no scaling and no cat cols, should pass through values unchanged
        assert torch.allclose(train_out.tensors[0], X_train)
        assert torch.allclose(val_out.tensors[0], X_val)

    def test_does_not_leak_val_into_train_stats(self):
        torch.manual_seed(0)
        # Train values in [0, 1], val values in [100, 101]. If val leaked into
        # scaler fit, train mean would shift far from 0.
        X_train = torch.rand(50, 2)
        X_val = 100 + torch.rand(10, 2)

        train_out, _ = preprocess_fold(
            _data_cfg(),
            TensorDataset(X_train, torch.zeros(50, 1)),
            TensorDataset(X_val, torch.zeros(10, 1)),
        )
        # Train mean must be ~0 if scaler was fit on train only
        assert abs(train_out.tensors[0].mean().item()) < 0.1
