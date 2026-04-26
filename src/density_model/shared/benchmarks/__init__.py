"""
Panel Benchmarks Package
------------------------
First-class benchmark models that produce a ``predictions.csv`` in the same
format as ``scripts/predict.py`` so the metrics script can compare them
side-by-side with the density model.

Classes
-------
BaseBenchmark
    Abstract interface every benchmark implements (fit + predict).
UnconditionalMeanBenchmark
    Per-asset historical mean; point forecast only (``sigma = NaN``).
AR1Benchmark
    Per-asset AR(1) via closed-form OLS; point forecast only
    (``sigma = NaN``).
GarchBenchmark
    Per-asset Constant-mean + GARCH(1,1) via the ``arch`` package; full
    Gaussian density forecast (mu + time-varying sigma).

Functions
---------
build_panel_benchmark
    Dispatch from a ``PanelBenchmarkConfig`` to a concrete benchmark.
run_panel_benchmark
    End-to-end: load train + predict CSVs, fit, write predictions.
"""

from __future__ import annotations

from density_model.shared.benchmarks.ar1 import AR1Benchmark
from density_model.shared.benchmarks.base import BaseBenchmark
from density_model.shared.benchmarks.build import build_panel_benchmark
from density_model.shared.benchmarks.garch import GarchBenchmark
from density_model.shared.benchmarks.run import run_panel_benchmark
from density_model.shared.benchmarks.unconditional_mean import UnconditionalMeanBenchmark

__all__ = [
    "AR1Benchmark",
    "BaseBenchmark",
    "GarchBenchmark",
    "UnconditionalMeanBenchmark",
    "build_panel_benchmark",
    "run_panel_benchmark",
]
