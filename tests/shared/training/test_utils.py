import pytest
import torch
import pandas as pd
from dlecosys.shared.training import MetricsTracker


class TestMetricsTracker:
    def test_update_and_last_log(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=0.5, acc=0.9)
        log = tracker.last_log()
        assert log["loss"] == pytest.approx(0.5)
        assert log["acc"] == pytest.approx(0.9)

    def test_update_accepts_scalar_tensor(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=torch.tensor(0.25))
        assert tracker.last_log()["loss"] == pytest.approx(0.25)

    def test_update_rejects_bool(self):
        with pytest.raises(TypeError):
            MetricsTracker().update_state(flag=True)

    def test_update_rejects_nan(self):
        with pytest.raises(ValueError):
            MetricsTracker().update_state(loss=float("nan"))

    def test_update_rejects_inf(self):
        with pytest.raises(ValueError):
            MetricsTracker().update_state(loss=float("inf"))

    def test_update_rejects_multielement_tensor(self):
        with pytest.raises(ValueError):
            MetricsTracker().update_state(loss=torch.tensor([0.1, 0.2]))

    def test_push_appends_rows(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=1.0)
        tracker.push(epoch=0)
        tracker.update_state(loss=0.5)
        tracker.push(epoch=1)
        assert len(tracker._rows) == 2

    def test_history_returns_dataframe(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=1.0)
        tracker.push(epoch=0)
        df = tracker.history
        assert isinstance(df, pd.DataFrame)
        assert "loss" in df.columns
        assert df["epoch"].iloc[0] == 0

    def test_history_empty_is_empty_dataframe(self):
        df = MetricsTracker().history
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_clear_current_preserves_history(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=1.0)
        tracker.push(epoch=0)
        tracker.clear_current()
        assert tracker.last_log() == {}
        assert len(tracker._rows) == 1

    def test_reset_state_clears_everything(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=1.0)
        tracker.push(epoch=0)
        tracker.reset_state()
        assert tracker.last_log() == {}
        assert tracker._rows == []

    def test_summary_min_max_avg(self):
        tracker = MetricsTracker()
        for i, v in enumerate([1.0, 2.0, 3.0]):
            tracker.update_state(loss=v)
            tracker.push(epoch=i)
        s = tracker.summary()
        assert s["loss"]["min"] == pytest.approx(1.0)
        assert s["loss"]["max"] == pytest.approx(3.0)
        assert s["loss"]["avg"] == pytest.approx(2.0)

    def test_summary_empty_returns_empty_dict(self):
        assert MetricsTracker().summary() == {}

    def test_later_update_overwrites_key(self):
        tracker = MetricsTracker()
        tracker.update_state(loss=1.0)
        tracker.update_state(loss=0.5)
        assert tracker.last_log()["loss"] == pytest.approx(0.5)
