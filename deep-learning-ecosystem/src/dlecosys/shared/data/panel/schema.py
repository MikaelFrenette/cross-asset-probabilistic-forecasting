"""
Panel Schema
------------
Typed model-facing batch and sample dataclasses for panel forecasting pipelines
operating on ``(B, ID, T, K)`` feature tensors and ``(B, ID, H, D_y)`` targets.

Classes
-------
PanelBatch
    Validated model-facing batch with continuous, dynamic-categorical, and
    static-categorical streams plus target and mask tensors.

PanelSample
    Per-sample torch dataset payload wrapping one forecast date into feature
    tensors, target, target mask, and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["PanelBatch", "PanelSample"]


@dataclass(frozen=True, slots=True)
class PanelSample:
    """
    Per-sample panel payload returned by torch dataset wrappers.

    Parameters
    ----------
    features : dict of str to torch.Tensor or None
        Model input tensors for one forecast date.
    target : torch.Tensor
        Target tensor for one forecast date.
    target_mask : torch.Tensor
        Target-availability mask for one forecast date.
    metadata : dict of str to Any
        Sample-level metadata such as forecast dates.
    """

    features: dict[str, torch.Tensor | None]
    target: torch.Tensor
    target_mask: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PanelBatch:
    """
    Model-facing batch schema for panel forecasting models.

    Parameters
    ----------
    X_continuous : torch.Tensor or None
        Continuous input tensor shaped ``(B, ID, T, D_cont)``.
    X_continuous_mask : torch.Tensor or None
        Continuous input mask shaped ``(B, ID, T, D_cont)``.
    X_cat_continuous : torch.Tensor or None
        Dynamic categorical input tensor shaped ``(B, ID, T, D_cat_dyn)``.
    X_cat_continuous_mask : torch.Tensor or None
        Dynamic categorical input mask shaped ``(B, ID, T, D_cat_dyn)``.
    X_cat_static : torch.Tensor or None
        Static categorical input tensor shaped ``(B, ID, T, D_cat_static)``.
    X_cat_static_mask : torch.Tensor or None
        Static categorical input mask shaped ``(B, ID, T, D_cat_static)``.
    y : torch.Tensor
        Target tensor shaped ``(B, ID, H, D_y)``.
    y_mask : torch.Tensor
        Target mask shaped ``(B, ID, H, D_y)``.
    id_index : list of Any
        Ordered identifier values for the ``ID`` axis.
    forecast_start_dates : Any
        Batch-aligned forecast start dates.
    forecast_end_dates : Any
        Batch-aligned forecast end dates.
    """

    X_continuous: torch.Tensor | None
    X_continuous_mask: torch.Tensor | None
    X_cat_continuous: torch.Tensor | None
    X_cat_continuous_mask: torch.Tensor | None
    X_cat_static: torch.Tensor | None
    X_cat_static_mask: torch.Tensor | None
    y: torch.Tensor
    y_mask: torch.Tensor
    id_index: list[Any]
    forecast_start_dates: Any
    forecast_end_dates: Any

    def __post_init__(self) -> None:
        self._validate_target_block()
        self._validate_feature_block(
            self.X_continuous, self.X_continuous_mask, "X_continuous", torch.float32
        )
        self._validate_feature_block(
            self.X_cat_continuous,
            self.X_cat_continuous_mask,
            "X_cat_continuous",
            torch.long,
        )
        self._validate_feature_block(
            self.X_cat_static,
            self.X_cat_static_mask,
            "X_cat_static",
            torch.long,
        )
        if len(self.id_index) != self.y.shape[1]:
            raise ValueError("id_index length must match the ID axis of y.")
        if len(self.forecast_start_dates) != self.y.shape[0]:
            raise ValueError("forecast_start_dates length must match the batch axis of y.")
        if len(self.forecast_end_dates) != self.y.shape[0]:
            raise ValueError("forecast_end_dates length must match the batch axis of y.")

    def as_training_tuple(
        self,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]:
        """Convert the batch schema to the trainer-facing tuple contract."""

        features = {
            "X_continuous": self.X_continuous,
            "X_continuous_mask": self.X_continuous_mask,
            "X_cat_continuous": self.X_cat_continuous,
            "X_cat_continuous_mask": self.X_cat_continuous_mask,
            "X_cat_static": self.X_cat_static,
            "X_cat_static_mask": self.X_cat_static_mask,
        }
        return features, self.y, self.y_mask

    @classmethod
    def from_training_batch(
        cls,
        *,
        features: dict[str, torch.Tensor | None],
        y: torch.Tensor,
        y_mask: torch.Tensor,
        id_index: list[Any] | None = None,
        forecast_start_dates: list[Any] | None = None,
        forecast_end_dates: list[Any] | None = None,
    ) -> PanelBatch:
        """Build a validated panel batch from the trainer-facing tuple contract."""

        batch_size, num_ids = y.shape[:2]
        resolved_id_index = id_index if id_index is not None else list(range(num_ids))
        resolved_forecast_start_dates = (
            forecast_start_dates
            if forecast_start_dates is not None
            else [None] * batch_size
        )
        resolved_forecast_end_dates = (
            forecast_end_dates if forecast_end_dates is not None else [None] * batch_size
        )
        return cls(
            X_continuous=features.get("X_continuous"),
            X_continuous_mask=features.get("X_continuous_mask"),
            X_cat_continuous=features.get("X_cat_continuous"),
            X_cat_continuous_mask=features.get("X_cat_continuous_mask"),
            X_cat_static=features.get("X_cat_static"),
            X_cat_static_mask=features.get("X_cat_static_mask"),
            y=y,
            y_mask=y_mask,
            id_index=resolved_id_index,
            forecast_start_dates=resolved_forecast_start_dates,
            forecast_end_dates=resolved_forecast_end_dates,
        )

    def _validate_target_block(self) -> None:
        if self.y.ndim != 4:
            raise ValueError("y must be four-dimensional with shape (B, ID, H, D_y).")
        if self.y_mask.ndim != 4:
            raise ValueError("y_mask must be four-dimensional with shape (B, ID, H, D_y).")
        if self.y.shape != self.y_mask.shape:
            raise ValueError("y and y_mask must share the same shape.")
        if self.y.dtype != torch.float32:
            raise TypeError("y must use torch.float32 dtype.")
        if self.y_mask.dtype != torch.bool:
            raise TypeError("y_mask must use torch.bool dtype.")

    def _validate_feature_block(
        self,
        values: torch.Tensor | None,
        mask: torch.Tensor | None,
        name: str,
        expected_dtype: torch.dtype,
    ) -> None:
        if values is None and mask is None:
            return
        if values is None or mask is None:
            raise ValueError(
                f"{name} and its mask must either both be present or both be None."
            )
        if values.ndim != 4:
            raise ValueError(f"{name} must be four-dimensional with shape (B, ID, T, D).")
        if mask.ndim != 4:
            raise ValueError(f"{name}_mask must be four-dimensional with shape (B, ID, T, D).")
        if values.shape != mask.shape:
            raise ValueError(f"{name} and its mask must share the same shape.")
        if values.shape[:2] != self.y.shape[:2]:
            raise ValueError(f"{name} must share the batch and ID axes of y.")
        if values.dtype != expected_dtype:
            raise TypeError(f"{name} must use dtype {expected_dtype}.")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name}_mask must use torch.bool dtype.")
