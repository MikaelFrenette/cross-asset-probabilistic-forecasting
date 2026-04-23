import pytest
import torch

from dlecosys.shared.config.builders import build_scheduler
from dlecosys.shared.config.schema import SchedulerSection


def _opt():
    lin = torch.nn.Linear(4, 2)
    return torch.optim.Adam(lin.parameters(), lr=0.01)


class TestBuildScheduler:
    def test_cosine_annealing(self):
        cfg = SchedulerSection(type="cosine_annealing", params={"T_max": 20})
        s = build_scheduler(cfg, _opt())
        assert isinstance(s, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step(self):
        cfg = SchedulerSection(type="step", params={"step_size": 10, "gamma": 0.5})
        s = build_scheduler(cfg, _opt())
        assert isinstance(s, torch.optim.lr_scheduler.StepLR)

    def test_exponential(self):
        cfg = SchedulerSection(type="exponential", params={"gamma": 0.9})
        s = build_scheduler(cfg, _opt())
        assert isinstance(s, torch.optim.lr_scheduler.ExponentialLR)

    def test_reduce_on_plateau(self):
        cfg = SchedulerSection(type="reduce_on_plateau", params={"factor": 0.5, "patience": 3})
        s = build_scheduler(cfg, _opt())
        assert isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_invalid_type_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SchedulerSection(type="one_cycle")

    def test_bad_param_propagates(self):
        # StepLR requires step_size; if we don't pass it, torch raises
        cfg = SchedulerSection(type="step", params={"gamma": 0.5})
        with pytest.raises(TypeError):
            build_scheduler(cfg, _opt())
