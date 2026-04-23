import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from dlecosys.shared.training.callbacks import TensorBoardCallback


class TestTensorBoardCallback:
    def test_writes_event_file(self, tmp_path):
        cb = TensorBoardCallback(log_dir=str(tmp_path / "tb"))
        cb.on_fit_start()
        cb.on_epoch_end(0, {"loss": 1.0, "val_loss": 0.9})
        cb.on_epoch_end(1, {"loss": 0.5, "val_loss": 0.6})
        cb.on_fit_end()

        # SummaryWriter writes files like events.out.tfevents.*
        files = list((tmp_path / "tb").iterdir())
        assert any(f.name.startswith("events.") for f in files)

    def test_skips_non_numeric_values(self, tmp_path):
        cb = TensorBoardCallback(log_dir=str(tmp_path / "tb"))
        cb.on_fit_start()
        cb.on_epoch_end(0, {"loss": 1.0, "note": "hello", "nested": {"x": 1}})  # must not raise
        cb.on_fit_end()

    def test_closes_writer_on_fit_end(self, tmp_path):
        cb = TensorBoardCallback(log_dir=str(tmp_path / "tb"))
        cb.on_fit_start()
        assert cb._writer is not None
        cb.on_fit_end()
        assert cb._writer is None
