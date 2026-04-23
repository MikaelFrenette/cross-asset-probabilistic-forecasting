import pytest
import torch

from dlecosys.shared.ensembling.aggregation import aggregate


class TestAggregate:
    def test_mean(self):
        preds = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])  # (2 est, 2 samples, 1)
        out = aggregate(preds, "mean")
        assert torch.allclose(out, torch.tensor([[2.0], [3.0]]))

    def test_median(self):
        preds = torch.tensor([[[1.0], [2.0]], [[5.0], [6.0]], [[3.0], [4.0]]])
        out = aggregate(preds, "median")
        assert torch.allclose(out, torch.tensor([[3.0], [4.0]]))

    def test_soft_vote_yields_class_labels(self):
        # 3 estimators, 2 samples, 3 classes
        preds = torch.tensor([
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[0.0, 10.0, 0.0], [0.0, 10.0, 0.0]],
        ])
        out = aggregate(preds, "soft_vote")
        # Sample 0: 2/3 estimators vote class 0 with huge margin → argmax = 0
        # Sample 1: unanimous class 1
        assert out.tolist() == [0, 1]

    def test_majority(self):
        preds = torch.tensor([
            [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
            [[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
            [[0.0, 5.0, 0.0], [0.0, 5.0, 0.0]],
        ])
        # argmax per estimator: [[0,1],[0,2],[1,1]]
        # Sample 0: {0,0,1} → majority = 0
        # Sample 1: {1,2,1} → majority = 1
        out = aggregate(preds, "majority")
        assert out.tolist() == [0, 1]

    def test_unknown_mode_raises(self):
        preds = torch.randn(2, 3, 1)
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate(preds, "geometric_mean")
