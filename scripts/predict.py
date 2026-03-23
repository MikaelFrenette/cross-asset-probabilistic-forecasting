"""
Prediction CLI
--------------
Command-line entrypoint for CSV-based next-session prediction runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density_model.config import load_predict_config
from density_model.inference import predict_from_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for prediction runs.

    Returns
    -------
    argparse.ArgumentParser
        Configured prediction CLI parser.
    """

    parser = argparse.ArgumentParser(description="Generate rolling forecasts from an extracted feature CSV.")
    parser.add_argument("--config", required=True, help="Path to the prediction YAML configuration file.")
    return parser


def main() -> None:
    """
    Run the prediction CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the prediction workflow and prints the output path.
    """

    args = build_argument_parser().parse_args()
    config = load_predict_config(args.config)
    output_path = predict_from_config(config=config)
    print(f"predictions_path={output_path}")


if __name__ == "__main__":
    main()
