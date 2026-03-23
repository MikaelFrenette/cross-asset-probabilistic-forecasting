"""
GARCH Benchmark Fit CLI
-----------------------
Command-line entrypoint for fitting the repository GARCH benchmark on a training
CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density_model.benchmarks import fit_garch_benchmark_from_config
from density_model.config import load_garch_benchmark_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for GARCH benchmark fitting.

    Returns
    -------
    argparse.ArgumentParser
        Configured GARCH benchmark fit CLI parser.
    """

    parser = argparse.ArgumentParser(description="Fit the GARCH benchmark from a training feature CSV.")
    parser.add_argument("--config", required=True, help="Path to the GARCH benchmark YAML configuration file.")
    return parser


def main() -> None:
    """
    Run the GARCH benchmark fit CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the benchmark fitting workflow and prints the artifact path.
    """

    args = build_argument_parser().parse_args()
    config = load_garch_benchmark_config(args.config)
    output_path = fit_garch_benchmark_from_config(config)
    print(f"garch_artifact_path={output_path}")


if __name__ == "__main__":
    main()
