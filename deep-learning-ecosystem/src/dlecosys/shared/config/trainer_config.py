"""
Trainer Configuration
----------------------
Pydantic-validated configuration model for BaseTrainer and its subclasses.

Classes
-------
BaseTrainerConfig
    Validates and stores all dependencies required to instantiate a trainer:
    model, optimizer, loss function, optional metrics, callbacks, verbosity,
    and strict-mode flag.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["BaseTrainerConfig"]


class BaseTrainerConfig(BaseModel):
    """
    Validated configuration for :class:`training.base.BaseTrainer`.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    optimizer : Optimizer
        Optimizer used for parameter updates.
    loss_fn : callable
        Loss function called as ``loss_fn(y_pred, y_true)``.
    metrics : dict[str, callable], optional
        Named metric functions called as ``fn(y_true=..., y_pred=...)``.
        Each must be callable.
    callbacks : list, optional
        Callback instances dispatched during the training lifecycle.
    verbose : {0, 1, 2}, default 1
        Verbosity level. 0 = silent, 1 = epoch summary, 2 = step-level bar.
    strict : bool, default True
        If True, raises when metrics are configured but ``info`` is missing
        ``y_true`` or ``y_pred``. If False, silently skips metric computation.
    device : torch.device or str, optional
        Device to move the model and batches to. Accepts strings such as
        ``"cpu"``, ``"cuda"``, or ``"cuda:0"``. If None, no device management
        is performed.
    amp : bool, default False
        Enable automatic mixed precision (AMP). When True, the training forward
        pass runs under ``torch.autocast`` and a ``GradScaler`` is used on CUDA
        devices. On CPU, autocast uses bfloat16; the scaler is disabled
        automatically since it is not supported outside CUDA.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    optimizer: Optimizer
    loss_fn: Callable
    metrics: Dict[str, Callable] = Field(default_factory=dict)
    callbacks: List[Any] = Field(default_factory=list)
    verbose: int = 1
    strict: bool = True
    device: Optional[torch.device] = None
    amp: bool = False
    grad_accumulation_steps: int = 1

    @field_validator("grad_accumulation_steps")
    @classmethod
    def validate_grad_accumulation_steps(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"grad_accumulation_steps must be >= 1; got {v!r}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: Any) -> nn.Module:
        if not isinstance(v, nn.Module):
            raise TypeError(f"model must be a torch.nn.Module; got {type(v).__name__}")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: Any) -> Optimizer:
        if not isinstance(v, Optimizer):
            raise TypeError(f"optimizer must be a torch.optim.Optimizer; got {type(v).__name__}")
        return v

    @field_validator("loss_fn")
    @classmethod
    def validate_loss_fn(cls, v: Any) -> Callable:
        if not callable(v):
            raise TypeError(f"loss_fn must be callable; got {type(v).__name__}")
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: Dict[str, Any]) -> Dict[str, Callable]:
        for name, fn in v.items():
            if not callable(fn):
                raise TypeError(
                    f"metrics['{name}'] must be callable; got {type(fn).__name__}"
                )
        return v

    @field_validator("verbose")
    @classmethod
    def validate_verbose(cls, v: int) -> int:
        if v not in {0, 1, 2}:
            raise ValueError(f"verbose must be 0, 1, or 2; got {v!r}")
        return v

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: Any) -> Optional[torch.device]:
        if v is None:
            return None
        if isinstance(v, torch.device):
            return v
        if isinstance(v, str):
            return torch.device(v)
        raise TypeError(f"device must be a str or torch.device; got {type(v).__name__}")
