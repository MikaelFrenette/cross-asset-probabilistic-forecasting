"""
GARCH Benchmark Predict CLI
---------------------------
Command-line entrypoint for rolling GARCH benchmark prediction from a prepared
feature CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density_model.benchmarks import predict_garch_from_config
from density_model.config import load_garch_benchmark_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for GARCH benchmark prediction.

    Returns
    -------
    argparse.ArgumentParser
        Configured GARCH benchmark prediction CLI parser.
    """

    parser = argparse.ArgumentParser(description="Generate rolling GARCH benchmark forecasts from a feature CSV.")
    parser.add_argument("--config", required=True, help="Path to the GARCH benchmark YAML configuration file.")
    parser.add_argument(
        "--dataset",
        required=False,
        help="Optional prediction dataset CSV override. Defaults to the benchmark config dataset path.",
    )
    return parser


def main() -> None:
    """
    Run the GARCH benchmark prediction CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the benchmark prediction workflow and prints the output path.
    """

    args = build_argument_parser().parse_args()
    config = load_garch_benchmark_config(args.config)
    output_path = predict_garch_from_config(config=config, dataset_path=args.dataset)
    print(f"garch_predictions_path={output_path}")


if __name__ == "__main__":
    main()
