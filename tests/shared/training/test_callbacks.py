import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from dlecosys.shared.training.callbacks import (
    CallbackList,
    EarlyStopping,
    GradNormCallback,
    LRSchedulerCallback,
    ModelCheckpoint,
)


class _FakeTrainer:
    """Minimal trainer stand-in for callback unit tests."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = nn.Linear(2, 1)
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{"lr": 0.01}]
        self.stop_training = False


class TestEarlyStopping:
    def _make(self, **kwargs):
        cb = EarlyStopping(**kwargs)
        cb.set_trainer(_FakeTrainer())
        cb.on_fit_start()
        return cb

    def test_triggers_after_patience(self):
        cb = self._make(monitor="val_loss", patience=2, mode="min")
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.6})
        cb.on_epoch_end(2, {"val_loss": 0.7})
        assert cb.trainer.stop_training is True
        assert cb.stopped_epoch == 2

    def test_does_not_trigger_before_patience(self):
        cb = self._make(monitor="val_loss", patience=3, mode="min")
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.6})
        assert cb.trainer.stop_training is False

    def test_resets_wait_on_improvement(self):
        cb = self._make(monitor="val_loss", patience=3, mode="min")
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.6})  # wait = 1
        cb.on_epoch_end(2, {"val_loss": 0.4})  # improved → wait = 0
        assert cb.wait == 0

    def test_warmup_epochs_ignored(self):
        # warmup=3: epochs 0-2 skipped; epoch 3 sets best=1.0 (no wait increment);
        # epochs 4-5 increment wait; patience=2 triggers at epoch 5.
        cb = self._make(monitor="val_loss", patience=2, mode="min", warmup=3)
        for i in range(6):
            cb.on_epoch_end(i, {"val_loss": 1.0})
        assert cb.trainer.stop_training is True

    def test_restore_best_weights(self):
        trainer = _FakeTrainer()
        cb = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
        cb.set_trainer(trainer)
        cb.on_fit_start()

        best_weight = trainer.model.weight.data.clone()
        cb.on_epoch_end(0, {"val_loss": 0.5})

        nn.init.constant_(trainer.model.weight, 99.0)
        cb.on_epoch_end(1, {"val_loss": 0.9})  # wait=1 → stop and restore

        assert torch.allclose(trainer.model.weight.data, best_weight)

    def test_missing_metric_is_ignored(self):
        cb = self._make(monitor="val_loss", patience=2, mode="min")
        cb.on_epoch_end(0, {"other": 0.5})
        assert cb.trainer.stop_training is False

    def test_max_mode(self):
        cb = self._make(monitor="val_acc", patience=2, mode="max")
        cb.on_epoch_end(0, {"val_acc": 0.8})
        cb.on_epoch_end(1, {"val_acc": 0.7})
        cb.on_epoch_end(2, {"val_acc": 0.6})
        assert cb.trainer.stop_training is True

    def test_on_fit_start_resets_state(self):
        cb = self._make(monitor="val_loss", patience=1, mode="min")
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_fit_start()
        assert cb.best is None
        assert cb.wait == 0


class TestModelCheckpoint:
    def test_saves_on_first_epoch(self, tmp_path):
        trainer = _FakeTrainer()
        cb = ModelCheckpoint(filepath=str(tmp_path / "best.pt"), monitor="val_loss", save_optimizer=False, verbose=False)
        cb.set_trainer(trainer)
        cb.on_fit_start()
        cb.on_epoch_end(0, {"val_loss": 0.5})
        assert (tmp_path / "best.pt").exists()

    def test_updates_best_on_improvement(self, tmp_path):
        trainer = _FakeTrainer()
        cb = ModelCheckpoint(filepath=str(tmp_path / "best.pt"), monitor="val_loss", save_optimizer=False, verbose=False)
        cb.set_trainer(trainer)
        cb.on_fit_start()
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.3})
        assert cb._best == pytest.approx(0.3)

    def test_does_not_save_on_regression(self, tmp_path):
        trainer = _FakeTrainer()
        path = tmp_path / "best.pt"
        cb = ModelCheckpoint(filepath=str(path), monitor="val_loss", save_optimizer=False, verbose=False)
        cb.set_trainer(trainer)
        cb.on_fit_start()
        cb.on_epoch_end(0, {"val_loss": 0.5})
        mtime = path.stat().st_mtime
        cb.on_epoch_end(1, {"val_loss": 0.9})
        assert path.stat().st_mtime == mtime

    def test_warmup_skips_checkpoint(self, tmp_path):
        trainer = _FakeTrainer()
        cb = ModelCheckpoint(filepath=str(tmp_path / "best.pt"), monitor="val_loss", warmup=2, save_optimizer=False, verbose=False)
        cb.set_trainer(trainer)
        cb.on_fit_start()
        cb.on_epoch_end(0, {"val_loss": 0.5})
        cb.on_epoch_end(1, {"val_loss": 0.4})
        assert not (tmp_path / "best.pt").exists()

    def test_checkpoint_contains_expected_keys(self, tmp_path):
        trainer = _FakeTrainer()
        path = str(tmp_path / "best.pt")
        cb = ModelCheckpoint(filepath=path, monitor="val_loss", save_optimizer=False, verbose=False)
        cb.set_trainer(trainer)
        cb.on_fit_start()
        cb.on_epoch_end(0, {"val_loss": 0.5})
        ckpt = torch.load(path, weights_only=False)
        assert {"model_state_dict", "epoch", "best_metric"}.issubset(ckpt.keys())


class TestLRSchedulerCallback:
    def test_steps_scheduler_each_epoch(self):
        scheduler = MagicMock()
        cb = LRSchedulerCallback(scheduler)
        cb.set_trainer(_FakeTrainer())
        cb.on_epoch_end(0, {})
        cb.on_epoch_end(1, {})
        assert scheduler.step.call_count == 2

    def test_passes_metric_value(self):
        scheduler = MagicMock()
        cb = LRSchedulerCallback(scheduler, monitor="val_loss")
        cb.set_trainer(_FakeTrainer())
        cb.on_epoch_end(0, {"val_loss": 0.4})
        scheduler.step.assert_called_once_with(0.4)

    def test_skips_step_when_metric_missing(self):
        scheduler = MagicMock()
        cb = LRSchedulerCallback(scheduler, monitor="val_loss")
        cb.set_trainer(_FakeTrainer())
        cb.on_epoch_end(0, {})
        scheduler.step.assert_not_called()


class TestCallbackList:
    def test_set_trainer_propagates_to_all(self):
        trainer = _FakeTrainer()
        cb1, cb2 = MagicMock(), MagicMock()
        cl = CallbackList([cb1, cb2])
        cl.set_trainer(trainer)
        cb1.set_trainer.assert_called_once_with(trainer)
        cb2.set_trainer.assert_called_once_with(trainer)

    def test_append_after_set_trainer_attaches(self):
        trainer = _FakeTrainer()
        cl = CallbackList()
        cl.set_trainer(trainer)
        cb = MagicMock()
        cl.append(cb)
        cb.set_trainer.assert_called_once_with(trainer)

    def test_exceptions_swallowed_by_default(self):
        class BrokenCallback:
            def set_trainer(self, t):
                pass

            def on_epoch_end(self, epoch, logs):
                raise RuntimeError("boom")

        cl = CallbackList([BrokenCallback()])
        cl.set_trainer(_FakeTrainer())
        cl.on_epoch_end(0, {})  # must not raise

    def test_exceptions_raised_when_configured(self):
        class BrokenCallback:
            def set_trainer(self, t):
                pass

            def on_epoch_end(self, epoch, logs):
                raise RuntimeError("boom")

        cl = CallbackList([BrokenCallback()], raise_errors=True)
        cl.set_trainer(_FakeTrainer())
        with pytest.raises(RuntimeError):
            cl.on_epoch_end(0, {})


class TestGradNormCallback:
    def test_logs_grad_norm_key(self):
        cb = GradNormCallback(log_key="grad_norm")
        trainer = _FakeTrainer()
        # Give the model a non-zero gradient.
        loss = trainer.model(torch.randn(2, 2)).sum()
        loss.backward()
        cb.set_trainer(trainer)
        # Attach a real MetricsTracker so the update doesn't crash.
        from dlecosys.shared.training.utils import MetricsTracker
        trainer.logger = MetricsTracker()
        cb.on_train_step_end(step=1, batch=None, outputs={}, logs={})
        assert "grad_norm" in trainer.logger.last_log()

    def test_grad_norm_is_positive(self):
        cb = GradNormCallback()
        trainer = _FakeTrainer()
        loss = trainer.model(torch.randn(2, 2)).sum()
        loss.backward()
        cb.set_trainer(trainer)
        from dlecosys.shared.training.utils import MetricsTracker
        trainer.logger = MetricsTracker()
        cb.on_train_step_end(step=1, batch=None, outputs={}, logs={})
        assert trainer.logger.last_log()["grad_norm"] > 0

    def test_zero_grad_logs_zero_norm(self):
        cb = GradNormCallback()
        trainer = _FakeTrainer()
        # Gradients are None by default (no backward called).
        cb.set_trainer(trainer)
        from dlecosys.shared.training.utils import MetricsTracker
        trainer.logger = MetricsTracker()
        cb.on_train_step_end(step=1, batch=None, outputs={}, logs={})
        assert trainer.logger.last_log()["grad_norm"] == pytest.approx(0.0)

    def test_custom_log_key(self):
        cb = GradNormCallback(log_key="my_norm")
        trainer = _FakeTrainer()
        cb.set_trainer(trainer)
        from dlecosys.shared.training.utils import MetricsTracker
        trainer.logger = MetricsTracker()
        cb.on_train_step_end(step=1, batch=None, outputs={}, logs={})
        assert "my_norm" in trainer.logger.last_log()
