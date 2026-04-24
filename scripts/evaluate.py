"""
evaluate.py — Compute held-out test metrics from predictions.csv.

Reads ``predictions.csv`` produced by ``predict.py`` (single-model or ensemble)
and writes ``evaluation.json`` alongside it. Chooses metrics based on
``cfg.data.task``:

    regression     → mae, mse, rmse
    classification → accuracy (hard labels) or accuracy on argmax (logits)

Usage
-----
    python scripts/evaluate.py --config <cfg>
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config import configure_logging, load_config
from dlecosys.shared.metrics import accuracy, mae, mse, rmse
from dlecosys.shared.run import EnsembleLayout, RunLayout

logger = logging.getLogger(__name__)


def _load_predictions(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    y_true, y_pred = [], []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:2] != ["y_true", "y_pred"]:
            raise ValueError(
                f"Unexpected header in {csv_path}: {header!r}. "
                f"Expected first two columns to be y_true, y_pred."
            )
        for row in reader:
            y_true.append(float(row[0]))
            y_pred.append(float(row[1]))
    return torch.tensor(y_true), torch.tensor(y_pred)


def _compute_regression(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_pred = y_pred.view_as(y_true)
    return {
        "n_samples": int(y_true.numel()),
        "mae": float(mae(y_true=y_true, y_pred=y_pred)),
        "mse": float(mse(y_true=y_true, y_pred=y_pred)),
        "rmse": float(rmse(y_true=y_true, y_pred=y_pred)),
    }


def _compute_classification(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    y_true = y_true.long()
    y_pred = y_pred.long()
    # accuracy() expects (y_true, y_pred) with class labels; csv stores floats,
    # so we cast both to long. predict.py's aggregation already reduced to class labels
    # for majority/soft_vote, or logits/regression values for mean/median — the user's
    # responsibility to pick the right aggregation for their task.
    return {
        "n_samples": int(y_true.numel()),
        "accuracy": float(accuracy(y_true=y_true, y_pred=y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a run's predictions.csv.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)

    if cfg.ensemble is not None:
        layout = EnsembleLayout(cfg.experiment.output_dir, cfg.experiment.name)
    else:
        layout = RunLayout(cfg.experiment.output_dir, cfg.experiment.name)

    if not layout.predictions_csv.exists():
        logger.error("predictions not found at %s — run predict.py first", layout.predictions_csv)
        sys.exit(1)

    y_true, y_pred = _load_predictions(layout.predictions_csv)

    if cfg.data.task == "regression":
        metrics = _compute_regression(y_true, y_pred)
    else:
        metrics = _compute_classification(y_true, y_pred)

    eval_path = layout.predictions_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump({"task": cfg.data.task, "metrics": metrics}, f, indent=2)

    print("=" * 72)
    print(f"Evaluation: {layout.root.name}")
    print("-" * 72)
    print(f"Task       : {cfg.data.task}")
    print(f"Predictions: {layout.predictions_csv}")
    for k, v in metrics.items():
        if isinstance(v, int):
            print(f"  {k:<12}: {v}")
        else:
            print(f"  {k:<12}: {v:.5f}")
    print(f"Written to : {eval_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
