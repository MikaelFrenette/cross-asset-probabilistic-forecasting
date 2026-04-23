"""
preprocess.py — Validate and preprocess tabular data for a pipeline run.

Reads a CSV from the path specified in the config, validates it, applies
configured preprocessing (scaling, tokenization), splits into train/val/test,
and saves the splits to the run output directory.

Scaling placement
-----------------
When ``cfg.tuning`` is set and ``cfg.tuning.scale_splits`` is true, this
script skips the scaler fit and writes the raw (unscaled, untokenized)
tensors. The tuning loop then applies transforms per fold. This is the
leak-free path for cross-validation-style tuning.

In all other cases (non-tuning runs, or ``scale_splits: false``), this
script fits the scaler + tokenizers on train and applies them to every
split at preprocess time — the legacy single-holdout behavior.

Usage
-----
    python scripts/preprocess.py --config configs/demo/regression.yaml
    python scripts/preprocess.py --config configs/demo/regression.yaml --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config import configure_logging, load_config
from dlecosys.shared.data import fit_transforms, load_tabular, split_tensors, validate_splits
from dlecosys.shared.reproducibility import seed_everything
from dlecosys.shared.run import DataPaths, EnsembleLayout, RunLayout, StudyLayout

logger = logging.getLogger(__name__)


def _load_xy(path: str, target_col: str, task: str):
    """Load tabular data into (X, y) tensors. Dispatches on file extension."""
    df = load_tabular(path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != target_col]
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    if task == "classification":
        y = torch.tensor(df[target_col].values, dtype=torch.long)
    else:
        y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and preprocess data for a run.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)
    seed_everything(cfg.experiment.seed, deterministic=cfg.experiment.deterministic)

    if cfg.tuning is not None and cfg.ensemble is not None:
        logger.error("config has both 'tuning' and 'ensemble' sections — choose one")
        sys.exit(1)

    if cfg.tuning is not None:
        layout: DataPaths = StudyLayout(cfg.experiment.output_dir, cfg.tuning.study_name)
    elif cfg.ensemble is not None:
        layout = EnsembleLayout(cfg.experiment.output_dir, cfg.experiment.name)
    else:
        layout = RunLayout(cfg.experiment.output_dir, cfg.experiment.name)
    layout.create(overwrite=args.overwrite)

    with open(layout.config_path, "w") as f:
        yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)

    X, y = _load_xy(cfg.data.path, cfg.data.target_col, cfg.data.task)

    if cfg.ensemble is not None:
        # Ensemble mode: no val split — the full pool is the train file.
        # Each estimator draws its own bootstrap + OOB inside run_ensemble.
        n = len(X)
        g = torch.Generator().manual_seed(cfg.experiment.seed)
        idx = torch.randperm(n, generator=g)
        n_test = int(n * cfg.data.test_size)
        splits = {
            "train": (X[idx[n_test:]], y[idx[n_test:]]),
            "test": (X[idx[:n_test]], y[idx[:n_test]]),
        }
    else:
        splits = split_tensors(X, y, cfg.data.val_size, cfg.data.test_size, cfg.experiment.seed)
    validate_splits(splits)

    # Skip fitting + applying transforms when tuning folds or ensemble bootstraps
    # will fit ephemeral per-fold / per-estimator transforms on demand.
    scale_at_preprocess = not (
        (cfg.tuning is not None and cfg.tuning.scale_splits)
        or cfg.ensemble is not None
    )

    if scale_at_preprocess:
        X_train = splits["train"][0]
        transforms = fit_transforms(
            X_train,
            categorical_cols=cfg.data.categorical_cols,
            scaling=cfg.data.scaling,
        )

        if transforms.tokenizers:
            layout.tokenizers_dir.mkdir(parents=True, exist_ok=True)
            for col_idx, tok in transforms.tokenizers.items():
                for split_name, (X_s, y_s) in splits.items():
                    col = tok.transform(X_s[:, col_idx].long().numpy())
                    X_s[:, col_idx] = torch.from_numpy(col).float()
                    splits[split_name] = (X_s, y_s)
                tok.save(str(layout.tokenizer_path(col_idx)))
                logger.info("tokenizer: col %d → %s", col_idx, layout.tokenizer_path(col_idx))

        if transforms.scaler is not None and transforms.cont_cols:
            for split_name, (X_s, y_s) in splits.items():
                X_s[:, transforms.cont_cols] = transforms.scaler.transform(
                    X_s[:, transforms.cont_cols]
                ).float()
                splits[split_name] = (X_s, y_s)
            transforms.scaler.save(str(layout.scaler_path))
            logger.info(
                "scaler: %s (cols %s) → %s",
                cfg.data.scaling,
                transforms.cont_cols,
                layout.scaler_path,
            )

        validate_splits(splits, tag="post-transform")
    else:
        if cfg.ensemble is not None:
            logger.info("ensemble mode — writing raw splits; each estimator scales its own bootstrap")
        else:
            logger.info("tuning with scale_splits=true — writing raw splits; fold loop will scale per fold")

    for split, (X_s, y_s) in splits.items():
        torch.save({"X": X_s, "y": y_s}, layout.data_path(split))
        logger.info("%s: X=%s  y=%s", split, tuple(X_s.shape), tuple(y_s.shape))

    logger.info("data written to %s", layout.data_dir)


if __name__ == "__main__":
    main()
