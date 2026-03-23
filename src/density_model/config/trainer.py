"""
Trainer Configuration
---------------------
Pydantic configuration objects for validating the public trainer interface and
associated PyTorch dependencies, including explicit device placement.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from density_model.training.callbacks import Callback
from density_model.training.inspectors import (
    LossFunctionInspector,
    MetricFunctionInspector,
    ModelInspector,
    OptimizerInspector,
)

__all__ = ["BaseTrainerConfig"]


class BaseTrainerConfig(BaseModel):
    """
    Pydantic configuration for the base training loop.

    Parameters
    ----------
    model : torch.nn.Module
        Model trained by the trainer.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates.
    loss_fn : Any
        Loss object accepting prediction and target tensors as positional arguments.
    metrics : dict of str to Any, default={}
        Metric objects keyed by public metric name and accepting two positional tensors.
    callbacks : list of Callback, default=[]
        Callback instances attached to the trainer lifecycle.
    device : str, default="cpu"
        Device identifier used for model and batch placement.
    verbose : int, default=1
        Verbosity level. Supported values are ``0``, ``1``, and ``2``.
    strict : bool, default=True
        Whether metric computation should raise when ``y_true`` or ``y_pred`` is missing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    model: Any
    optimizer: Any
    loss_fn: Any
    metrics: dict[str, Any] = Field(default_factory=dict)
    callbacks: list[Callback] = Field(default_factory=list)
    device: str = "cpu"
    verbose: int = Field(default=1, ge=0, le=2)
    strict: bool = True

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: Any) -> Any:
        """
        Validate the configured PyTorch model.

        Parameters
        ----------
        value : Any
            Model candidate.

        Returns
        -------
        Any
            Validated model object.
        """

        ModelInspector.validate(value, name="model")
        return value

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, value: Any) -> Any:
        """
        Validate the configured optimizer.

        Parameters
        ----------
        value : Any
            Optimizer candidate.

        Returns
        -------
        Any
            Validated optimizer object.
        """

        OptimizerInspector.validate(value, name="optimizer")
        return value

    @field_validator("loss_fn")
    @classmethod
    def validate_loss_fn(cls, value: Any) -> Any:
        """
        Validate the configured loss object.

        Parameters
        ----------
        value : Any
            Loss candidate.

        Returns
        -------
        Any
            Validated loss object.
        """

        LossFunctionInspector.validate(value, name="loss_fn")
        return value

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, Any]) -> dict[str, Any]:
        """
        Validate configured metric objects.

        Parameters
        ----------
        value : dict of str to Any
            Metric mapping keyed by public metric name.

        Returns
        -------
        dict of str to Any
            Validated metric mapping.
        """

        for name, metric in value.items():
            if not isinstance(name, str) or not name:
                raise TypeError("Metric names must be non-empty strings.")
            MetricFunctionInspector.validate(metric, name=f"metrics[{name!r}]")
        return value

    @field_validator("callbacks")
    @classmethod
    def validate_callbacks(cls, value: list[Callback]) -> list[Callback]:
        """
        Validate configured callback instances.

        Parameters
        ----------
        value : list of Callback
            Callback sequence attached to the trainer.

        Returns
        -------
        list of Callback
            Validated callback sequence.
        """

        for callback in value:
            if not isinstance(callback, Callback):
                raise TypeError("Each callback must inherit from density_model.training.callbacks.Callback.")
        return value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """
        Validate the configured torch device string.

        Parameters
        ----------
        value : str
            Requested device identifier.

        Returns
        -------
        str
            Validated device identifier.

        Raises
        ------
        ValueError
            If the device string is invalid or unavailable.
        """

        try:
            device = torch.device(value)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Invalid torch device: {value!r}.") from error

        if device.type == "cuda" and not torch.cuda.is_available():
            details = [
                "CUDA device requested but CUDA is not available in the active PyTorch runtime.",
                f"torch.__version__={torch.__version__}",
                f"torch.version.cuda={torch.version.cuda}",
                f"torch.cuda.device_count()={torch.cuda.device_count()}",
            ]
            if os.environ.get("WSL_DISTRO_NAME"):
                if not os.path.exists("/dev/dxg"):
                    details.append(
                        "WSL GPU passthrough device `/dev/dxg` is missing, so this Linux environment cannot access the GPU."
                    )
                else:
                    details.append("WSL environment detected.")
            raise ValueError(" ".join(details))
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS device requested but MPS is not available.")
        return str(device)
