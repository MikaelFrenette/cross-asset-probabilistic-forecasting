"""
Generate Metrics Report CLI
---------------------------
Command-line entrypoint for generating conditional-mean RMSE ratio reports
relative to a zero-return benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from density_model.evaluation.metrics_report import generate_metrics_report


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for the metrics report workflow.

    Returns
    -------
    argparse.ArgumentParser
        Configured metrics report CLI parser.
    """

    parser = argparse.ArgumentParser(
        description="Generate RMSE ratio and coverage reports against a zero-return benchmark."
    )
    parser.add_argument("--train-csv", required=True, help="Training feature CSV path.")
    parser.add_argument("--dataset-csv", required=True, help="Prediction/test feature CSV path.")
    parser.add_argument("--validation-csv", default=None, help="Optional validation feature CSV path.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        dest="models",
        help="Model spec in the form name=path/to/predictions.csv. Repeat once per model.",
    )
    parser.add_argument("--target-column", default="return", help="Target column used for evaluation.")
    parser.add_argument("--date-column", default="date", help="Date column in the feature CSVs.")
    parser.add_argument("--id-column", default="asset_id", help="Identifier column in the feature CSVs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the metrics report.")
    return parser


def main() -> None:
    """
    Run the metrics report CLI from parsed command-line arguments.

    Returns
    -------
    None
        This function executes the metrics-report workflow and prints the
        output directory.
    """

    args = build_argument_parser().parse_args()
    output_dir = generate_metrics_report(
        train_csv=args.train_csv,
        dataset_csv=args.dataset_csv,
        model_specs=args.models,
        target_column=args.target_column,
        date_column=args.date_column,
        id_column=args.id_column,
        validation_csv=args.validation_csv,
        output_dir=args.output_dir,
    )
    print(f"metrics_report_dir={output_dir}")


if __name__ == "__main__":
    main()
