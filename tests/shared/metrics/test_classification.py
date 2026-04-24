import pytest
import torch
from dlecosys.shared.metrics import accuracy, binary_accuracy


class TestAccuracy:
    def test_perfect(self):
        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert accuracy(y_true=y_true, y_pred=y_pred).item() == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = torch.tensor([0, 0, 0])
        y_pred = torch.tensor([[0.0, 1.0, 0.0]] * 3)
        assert accuracy(y_true=y_true, y_pred=y_pred).item() == pytest.approx(0.0)

    def test_half_correct(self):
        y_true = torch.tensor([0, 1])
        y_pred = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # both predict class 0
        assert accuracy(y_true=y_true, y_pred=y_pred).item() == pytest.approx(0.5)


class TestBinaryAccuracy:
    def test_perfect_from_logits(self):
        y_true = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
        y_pred = torch.tensor([10.0, -10.0, 10.0, -10.0])
        assert binary_accuracy(y_true=y_true, y_pred=y_pred).item() == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true = torch.tensor([1, 1], dtype=torch.float32)
        y_pred = torch.tensor([-10.0, -10.0])
        assert binary_accuracy(y_true=y_true, y_pred=y_pred).item() == pytest.approx(0.0)

    def test_custom_threshold(self):
        # sigmoid(0.6) ≈ 0.646, which is above threshold=0.6 → predicted as 1
        y_true = torch.tensor([1], dtype=torch.float32)
        y_pred = torch.tensor([0.6])
        assert binary_accuracy(y_true=y_true, y_pred=y_pred, threshold=0.6).item() == pytest.approx(1.0)
