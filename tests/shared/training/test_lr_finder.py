import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dlecosys.shared.training import LRFinder, LRFinderResult, Trainer


def _make_trainer(seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(model=model, optimizer=optimizer, loss_fn=nn.MSELoss(), verbose=0)


def _make_loader(n=40, features=4, batch_size=8, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, features)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


class TestLRFinderResult:
    def test_suggest_valley_returns_float(self):
        result = LRFinderResult(
            lrs=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            losses=[1.0, 0.8, 0.5, 0.6, 1.2],
        )
        lr = result.suggest("valley")
        assert isinstance(lr, float)
        assert lr > 0

    def test_suggest_min_returns_lr_before_minimum(self):
        result = LRFinderResult(
            lrs=[1e-4, 1e-3, 1e-2, 1e-1],
            losses=[0.9, 0.7, 0.4, 0.8],
        )
        # minimum is at index 2 (lr=1e-2), expect index 1 (lr=1e-3)
        lr = result.suggest("min")
        assert lr == pytest.approx(1e-3)

    def test_suggest_invalid_method_raises(self):
        result = LRFinderResult(lrs=[1e-3, 1e-2], losses=[0.5, 0.4])
        with pytest.raises(ValueError):
            result.suggest("unknown")

    def test_suggest_too_few_points_raises(self):
        result = LRFinderResult(lrs=[1e-3], losses=[0.5])
        with pytest.raises(RuntimeError):
            result.suggest()


class TestLRFinder:
    def test_run_returns_result_with_data(self):
        trainer = _make_trainer()
        finder = LRFinder(trainer)
        result = finder.run(_make_loader(), num_steps=10)
        assert isinstance(result, LRFinderResult)
        assert len(result.lrs) > 0
        assert len(result.lrs) == len(result.losses)

    def test_lrs_are_ascending(self):
        trainer = _make_trainer()
        finder = LRFinder(trainer)
        result = finder.run(_make_loader(), start_lr=1e-5, end_lr=1.0, num_steps=20)
        assert all(a < b for a, b in zip(result.lrs, result.lrs[1:]))

    def test_model_weights_restored_after_run(self):
        trainer = _make_trainer()
        weights_before = trainer.model.weight.data.clone()
        LRFinder(trainer).run(_make_loader(), num_steps=10)
        assert torch.allclose(trainer.model.weight.data, weights_before)

    def test_optimizer_lr_restored_after_run(self):
        trainer = _make_trainer()
        original_lr = trainer.optimizer.param_groups[0]["lr"]
        LRFinder(trainer).run(_make_loader(), num_steps=10)
        assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)

    def test_accumulation_steps_restored_after_run(self):
        trainer = _make_trainer()
        trainer.cfg.grad_accumulation_steps = 4
        LRFinder(trainer).run(_make_loader(), num_steps=10)
        assert trainer.cfg.grad_accumulation_steps == 4

    def test_diverge_stops_early(self):
        trainer = _make_trainer()
        result = LRFinder(trainer).run(
            _make_loader(), start_lr=1.0, end_lr=1e6, num_steps=50, diverge_threshold=2.0
        )
        assert len(result.lrs) < 50

    def test_dataloader_cycled_when_exhausted(self):
        # Use a loader with fewer samples than num_steps × batch_size
        trainer = _make_trainer()
        small_loader = _make_loader(n=8, batch_size=8)
        result = LRFinder(trainer).run(small_loader, num_steps=20)
        assert len(result.lrs) > 0

    def test_suggest_after_run(self):
        trainer = _make_trainer()
        result = LRFinder(trainer).run(_make_loader(), num_steps=30)
        lr = result.suggest()
        assert isinstance(lr, float)
        assert lr > 0
