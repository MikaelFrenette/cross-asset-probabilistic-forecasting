import pytest
import torch
from dlecosys.shared.metrics import mae, mse, rmse, mape


def _t(*vals):
    return torch.tensor(vals, dtype=torch.float32)


class TestMae:
    def test_perfect_predictions(self):
        y = _t(1.0, 2.0, 3.0)
        assert mae(y_true=y, y_pred=y).item() == pytest.approx(0.0)

    def test_constant_error(self):
        assert mae(y_true=_t(0.0, 0.0, 0.0), y_pred=_t(1.0, 1.0, 1.0)).item() == pytest.approx(1.0)

    def test_mixed_signs(self):
        # |0 - 2| + |0 - (-2)| / 2 = 2.0
        assert mae(y_true=_t(2.0, -2.0), y_pred=_t(0.0, 0.0)).item() == pytest.approx(2.0)


class TestMse:
    def test_perfect_predictions(self):
        y = _t(1.0, 2.0, 3.0)
        assert mse(y_true=y, y_pred=y).item() == pytest.approx(0.0)

    def test_known_value(self):
        # (2^2 + 4^2) / 2 = (4 + 16) / 2 = 10
        assert mse(y_true=_t(0.0, 0.0), y_pred=_t(2.0, 4.0)).item() == pytest.approx(10.0)


class TestRmse:
    def test_equals_sqrt_mse(self):
        y_true = _t(1.0, 2.0, 3.0, 4.0)
        y_pred = _t(1.5, 2.5, 2.5, 3.5)
        expected = mse(y_true=y_true, y_pred=y_pred).sqrt().item()
        assert rmse(y_true=y_true, y_pred=y_pred).item() == pytest.approx(expected)

    def test_perfect_predictions(self):
        y = _t(1.0, 2.0)
        assert rmse(y_true=y, y_pred=y).item() == pytest.approx(0.0)


class TestMape:
    def test_perfect_predictions(self):
        y = _t(1.0, 2.0, 3.0)
        assert mape(y_true=y, y_pred=y).item() == pytest.approx(0.0)

    def test_known_value(self):
        # |110 - 100| / 100 = 0.1
        assert mape(y_true=_t(100.0), y_pred=_t(110.0)).item() == pytest.approx(0.1, abs=1e-5)

    def test_eps_prevents_division_by_zero(self):
        result = mape(y_true=_t(0.0), y_pred=_t(1.0))
        assert torch.isfinite(result)
