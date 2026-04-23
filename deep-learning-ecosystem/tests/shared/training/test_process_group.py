"""Tests for process_group utilities when torch.distributed is not initialised."""

import os

import pytest

from dlecosys.shared.training.process_group import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
    teardown,
)


class TestProcessGroupDefaults:
    """All helpers must return safe single-process defaults when dist is not init."""

    def test_get_rank_returns_zero(self):
        assert get_rank() == 0

    def test_get_world_size_returns_one(self):
        assert get_world_size() == 1

    def test_is_main_process_returns_true(self):
        assert is_main_process() is True

    def test_teardown_is_safe_when_not_initialised(self):
        teardown()  # must not raise

    def test_get_local_rank_from_env(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert get_local_rank() == 2

    def test_get_local_rank_default_zero(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert get_local_rank() == 0
