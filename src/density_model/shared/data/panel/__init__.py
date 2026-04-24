"""
Panel Data Package
------------------
Panel-shaped data utilities for (B, ID, T, K) forecasting pipelines: schema
objects, feature planning, time-series inspection, vectorized panel assembly,
PyTorch dataset wrappers, market calendars, and the Yahoo-based panel builder.

Classes
-------
Re-exported from submodules for convenience.
"""

from __future__ import annotations

from density_model.shared.data.panel.calendar import MarketCalendar, XNYSCalendar
from density_model.shared.data.panel.config import (
    FeatureInputConfig,
    ForecastingConfig,
    TimeSeriesConfig,
    VectorizedPanelConfig,
)
from density_model.shared.data.panel.dataset import BasePanelTorchDataset, VectorizedPanelTorchDataset
from density_model.shared.data.panel.features import FeatureGroups, FeaturePlanner, FeatureResolver
from density_model.shared.data.panel.inspectors import (
    ForecastingInspector,
    FrameBackend,
    TimeSeriesInspector,
)
from density_model.shared.data.panel.schema import PanelBatch, PanelSample
from density_model.shared.data.panel.sources import (
    YahooDailyReturnsLoader,
    YahooDailyReturnsRequest,
    YahooVolatilityPanelBuilder,
)
from density_model.shared.data.panel.splitter import (
    BasePanelSplitter,
    PanelExpandingWindowSplitter,
    PanelHoldoutSplitter,
    PanelWalkForwardSplitter,
)
from density_model.shared.data.panel.vectorizer import BaseTimeSeriesDataset, VectorizedPanelDataset

__all__ = [
    "BasePanelSplitter",
    "BasePanelTorchDataset",
    "BaseTimeSeriesDataset",
    "FeatureGroups",
    "FeatureInputConfig",
    "FeaturePlanner",
    "FeatureResolver",
    "ForecastingConfig",
    "ForecastingInspector",
    "FrameBackend",
    "MarketCalendar",
    "PanelBatch",
    "PanelExpandingWindowSplitter",
    "PanelHoldoutSplitter",
    "PanelSample",
    "PanelWalkForwardSplitter",
    "TimeSeriesConfig",
    "TimeSeriesInspector",
    "VectorizedPanelConfig",
    "VectorizedPanelDataset",
    "VectorizedPanelTorchDataset",
    "XNYSCalendar",
    "YahooDailyReturnsLoader",
    "YahooDailyReturnsRequest",
    "YahooVolatilityPanelBuilder",
]
