"""
Process Group Utilities
-----------------------
Thin wrappers over torch.distributed for DDP setup and rank queries.

Functions
---------
setup
    Initialise the default process group from torchrun environment variables.
teardown
    Destroy the process group if initialised.
get_rank
    Global rank of this process (0 if not in distributed mode).
get_local_rank
    Local CUDA device index for this process.
get_world_size
    Total number of processes (1 if not in distributed mode).
is_main_process
    True when this process is rank 0.
"""

from __future__ import annotations

import os

import torch.distributed as dist

__all__ = [
    "setup",
    "teardown",
    "get_rank",
    "get_local_rank",
    "get_world_size",
    "is_main_process",
]


def setup(backend: str) -> None:
    """Initialise the default process group using torchrun env vars."""
    dist.init_process_group(backend=backend, init_method="env://")


def teardown() -> None:
    """Destroy the process group if it was initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Return global rank; 0 when not in distributed mode."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_local_rank() -> int:
    """Return local CUDA device index from LOCAL_RANK env var; 0 if absent."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    """Return total number of processes; 1 when not in distributed mode."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    """Return True for rank 0 (the only process when not in distributed mode)."""
    return get_rank() == 0
