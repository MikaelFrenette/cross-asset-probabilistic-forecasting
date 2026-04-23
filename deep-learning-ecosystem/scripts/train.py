"""
train.py — Train a model for a pipeline run.

Usage
-----
    # Single-GPU / CPU
    python scripts/train.py --config configs/demo/regression.yaml

    # Multi-GPU (DDP) — set distributed.enabled: true in config
    torchrun --nproc_per_node=4 scripts/train.py --config configs/myproject/exp.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dlecosys.models  # noqa: F401 — registers bundled models with the factory

from dlecosys.shared.config import configure_logging, load_config
from dlecosys.shared.run import RunLayout
from dlecosys.shared.training import run_training

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model for a run.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoints/logs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)

    if cfg.ensemble is not None:
        logger.error(
            "config has an 'ensemble' section — use scripts/ensemble.py, not train.py"
        )
        sys.exit(1)

    layout = RunLayout(cfg.experiment.output_dir, cfg.experiment.name)

    if not layout.data_dir.exists():
        logger.error("data not found at %s — run preprocess.py first", layout.data_dir)
        sys.exit(1)

    if layout.checkpoint_path.exists() and not args.overwrite:
        logger.error("checkpoint already exists at %s — use --overwrite to retrain", layout.checkpoint_path)
        sys.exit(1)

    run_training(cfg, layout, summary_extras={"run_name": cfg.experiment.name})


if __name__ == "__main__":
    main()
