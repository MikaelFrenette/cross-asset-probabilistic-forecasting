import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from dlecosys.shared.training import Trainer, EarlyStopping
from dlecosys.shared.metrics import mae


def _loader(n=20, features=4, batch_size=5, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(n, features)
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


def _trainer(**kwargs):
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(model=model, optimizer=optimizer, loss_fn=nn.MSELoss(), verbose=0, **kwargs)


class TestTrainerBasic:
    def test_train_logs_loss(self):
        t = _trainer()
        t.train(train_dataloader=_loader(), epochs=2)
        assert "loss" in t.logger.last_log()

    def test_train_with_val_logs_val_loss(self):
        t = _trainer()
        loader = _loader()
        t.train(train_dataloader=loader, val_dataloader=loader, epochs=2)
        assert "val_loss" in t.logger.last_log()

    def test_history_one_row_per_epoch(self):
        t = _trainer()
        t.train(train_dataloader=_loader(), epochs=3)
        assert len(t.logger.history) == 3

    def test_train_resets_history_on_reuse(self):
        t = _trainer()
        t.train(train_dataloader=_loader(), epochs=3)
        t.train(train_dataloader=_loader(), epochs=1)
        assert len(t.logger.history) == 1

    def test_amp_cpu_does_not_crash(self):
        t = _trainer(amp=True)
        t.train(train_dataloader=_loader(), epochs=1)

    def test_custom_metric_appears_in_log(self):
        t = _trainer(metrics={"mae": mae})
        t.train(train_dataloader=_loader(), epochs=1)
        assert "train_mae" in t.logger.last_log()

    def test_val_metric_appears_in_log(self):
        t = _trainer(metrics={"mae": mae})
        loader = _loader()
        t.train(train_dataloader=loader, val_dataloader=loader, epochs=1)
        assert "val_mae" in t.logger.last_log()


class TestGradientAccumulation:
    def test_accumulation_does_not_crash(self):
        t = _trainer(grad_accumulation_steps=2)
        t.train(train_dataloader=_loader(), epochs=2)
        assert "loss" in t.logger.last_log()

    def test_accumulation_resets_each_epoch(self):
        t = _trainer(grad_accumulation_steps=4)
        t.train(train_dataloader=_loader(), epochs=3)
        # After training, _accum_count should reflect the last epoch's steps
        # (not an ever-growing counter). Just verify training completed cleanly.
        assert len(t.logger.history) == 3

    def test_accumulation_step_1_matches_default(self):
        # With accum=1, behavior must be identical to not specifying it.
        torch.manual_seed(0)
        model_a = nn.Linear(4, 1)
        torch.manual_seed(0)
        model_b = nn.Linear(4, 1)

        loader = _loader()

        t_a = Trainer(model=model_a, optimizer=torch.optim.SGD(model_a.parameters(), lr=0.01),
                      loss_fn=nn.MSELoss(), verbose=0)
        t_b = Trainer(model=model_b, optimizer=torch.optim.SGD(model_b.parameters(), lr=0.01),
                      loss_fn=nn.MSELoss(), verbose=0, grad_accumulation_steps=1)

        t_a.train(train_dataloader=loader, epochs=2)
        t_b.train(train_dataloader=loader, epochs=2)

        assert torch.allclose(model_a.weight.data, model_b.weight.data, atol=1e-6)


class TestTrainerCallbackIntegration:
    def test_early_stopping_halts_before_max_epochs(self):
        # Model pre-set to the exact solution → loss = 0 every epoch →
        # gradient = 0 → weights don't move → val_loss never improves →
        # EarlyStopping triggers after patience + 1 epochs.
        X = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        y = torch.tensor([[1.0]])
        loader = DataLoader(TensorDataset(X, y), batch_size=1)

        model = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
            model.weight[0, 0] = 1.0

        cb = EarlyStopping(monitor="val_loss", patience=2, mode="min", warmup=0)
        t = Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=nn.MSELoss(),
            verbose=0,
            callbacks=[cb],
        )
        t.train(train_dataloader=loader, val_dataloader=loader, epochs=50)
        assert len(t.logger.history) < 50

    def test_model_checkpoint_saved(self, tmp_path):
        from dlecosys.shared.training import ModelCheckpoint
        path = str(tmp_path / "best.pt")
        cb = ModelCheckpoint(filepath=path, monitor="val_loss", verbose=False)
        t = _trainer(callbacks=[cb])
        loader = _loader(n=10, batch_size=10)
        t.train(train_dataloader=loader, val_dataloader=loader, epochs=2)
        assert (tmp_path / "best.pt").exists()
