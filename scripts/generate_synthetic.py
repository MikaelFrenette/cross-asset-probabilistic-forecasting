"""
generate_synthetic.py — Generate synthetic CSV datasets for demo runs.

Reads generation parameters from configs/demo/synthetic.yaml and writes
one CSV per task to the configured output paths. This script is not part
of the training pipeline — it is a one-time data generation utility.

Usage
-----
    python scripts/generate_synthetic.py --task regression
    python scripts/generate_synthetic.py --task classification
    python scripts/generate_synthetic.py --task regression --config configs/demo/synthetic.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config.schema import SyntheticConfig
from dlecosys.shared.data.synthetic import make_classification_data, make_regression_data
from dlecosys.shared.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CSV data for a demo task.")
    parser.add_argument("--task", required=True, choices=["regression", "classification"])
    parser.add_argument(
        "--config",
        default="configs/demo/synthetic.yaml",
        help="Path to synthetic data config YAML.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)
    cfg = SyntheticConfig(**raw)

    section = getattr(cfg, args.task)
    if section is None:
        print(f"No '{args.task}' section found in {args.config}.")
        sys.exit(1)

    seed_everything(section.seed)

    if section.task == "regression":
        X_all, y_all = make_regression_data(
            section.n_samples, section.n_features, section.noise, section.seed
        )
    else:
        X_all, y_all = make_classification_data(
            section.n_samples, section.n_features, section.n_classes, section.noise, section.seed
        )

    n_features = X_all.shape[1]
    feature_cols = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame(X_all.numpy(), columns=feature_cols)
    df[section.target_col] = y_all.numpy().flatten()

    output_path = Path(section.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("saved %d rows × %d features → %s", len(df), n_features, output_path)


if __name__ == "__main__":
    main()
