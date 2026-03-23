"""
Generate Plot CLI
-----------------
Command-line entrypoint for generating the repository forecast report plots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density_model.config import build_plot_config
from density_model.evaluation import generate_plot_from_config


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for plotting runs.

    Returns
    -------
    argparse.ArgumentParser
        Configured plotting CLI parser.
    """

    parser = argparse.ArgumentParser(description="Generate repository forecast report plots.")
    parser.add_argument("--manifest-path", required=True, help="Frozen training manifest path.")
    parser.add_argument("--predictions-csv", required=True, help="Model prediction CSV path.")
    parser.add_argument("--dataset-csv", required=True, help="Realized dataset CSV path.")
    parser.add_argument("--output-dir", required=True, help="Plot output directory.")
    parser.add_argument("--date-column", default="date", help="Realized dataset date column name.")
    parser.add_argument("--id-column", default="asset_id", help="Realized dataset identifier column name.")
    return parser


def main() -> None:
    """
    Run the plotting CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the plotting workflow and prints the output directory.
    """

    args = build_argument_parser().parse_args()
    config = build_plot_config(
        manifest_path=args.manifest_path,
        predictions_csv=args.predictions_csv,
        benchmark_predictions_csv=None,
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
        date_column=args.date_column,
        id_column=args.id_column,
    )
    output_dir = generate_plot_from_config(config=config)
    print(f"plots_dir={output_dir}")


if __name__ == "__main__":
    main()
