"""
Benchmark Package
-----------------
Optional benchmark workflows and artifact helpers.
"""

from density_model.benchmarks.garch import (
    GarchAssetArtifact,
    GarchBenchmarkArtifact,
    fit_garch_benchmark_from_config,
    predict_garch_from_config,
)

__all__ = [
    "GarchAssetArtifact",
    "GarchBenchmarkArtifact",
    "fit_garch_benchmark_from_config",
    "predict_garch_from_config",
]
