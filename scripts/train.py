"""
Training CLI
------------
Command-line entrypoint for CSV-based training runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density_model.config import load_train_config
from density_model.workflows import train_from_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for training runs.

    Returns
    -------
    argparse.ArgumentParser
        Configured training CLI parser.
    """

    parser = argparse.ArgumentParser(description="Train the density-model forecasting stack from an extracted CSV.")
    parser.add_argument("--config", required=True, help="Path to the training YAML configuration file.")
    parser.add_argument(
        "--resume-from",
        required=False,
        help="Optional checkpoint path used to resume model and optimizer state.",
    )
    return parser


def main() -> None:
    """
    Run the training CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the training workflow and prints the artifact path.
    """

    args = build_argument_parser().parse_args()
    config = load_train_config(args.config)
    artifact_dir = train_from_config(config, resume_from=args.resume_from)
    print(f"artifacts_dir={artifact_dir}")


if __name__ == "__main__":
    main()
