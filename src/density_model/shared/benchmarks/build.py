"""
Benchmark Builder
-----------------
Dispatch from a :class:`PanelBenchmarkConfig` to a concrete benchmark class.

Functions
---------
build_panel_benchmark
    Return a fresh ``BaseBenchmark`` instance matching ``cfg.type``.
"""

from __future__ import annotations

from density_model.shared.benchmarks.ar1 import AR1Benchmark
from density_model.shared.benchmarks.base import BaseBenchmark
from density_model.shared.benchmarks.garch import GarchBenchmark
from density_model.shared.benchmarks.unconditional_mean import UnconditionalMeanBenchmark
from density_model.shared.config.panel_schema import PanelBenchmarkConfig

__all__ = ["build_panel_benchmark"]


def build_panel_benchmark(cfg: PanelBenchmarkConfig) -> BaseBenchmark:
    if cfg.type == "unconditional_mean":
        return UnconditionalMeanBenchmark()
    if cfg.type == "ar1":
        return AR1Benchmark()
    if cfg.type == "garch":
        return GarchBenchmark()
    raise ValueError(f"Unsupported benchmark type {cfg.type!r}.")
