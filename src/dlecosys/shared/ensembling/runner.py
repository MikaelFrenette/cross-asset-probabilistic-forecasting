"""
Ensemble Runner
---------------
Orchestrates training of N estimators for a bagging ensemble.

For each estimator the runner:
    1. Generates bootstrap + OOB indices via the ensemble's hook
    2. Slices the raw pool into (bootstrap, OOB) TensorDatasets
    3. Fits ephemeral tokenizers + scaler on the bootstrap
    4. Persists those transforms under the estimator's directory (for inference)
    5. Calls ``run_training(train=bootstrap, val=OOB)`` — EarlyStopping and
       ModelCheckpoint therefore target OOB performance
    6. Renders a TF-Tuner-style header before each fit (clears console)

Functions
---------
run_ensemble
    Train every estimator configured by ``cfg.ensemble``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from typing import Dict, List

import torch
import yaml
from torch.utils.data import TensorDataset

from dlecosys.shared.config.schema import PipelineConfig
from dlecosys.shared.data import apply_transforms, fit_transforms
from dlecosys.shared.ensembling.base import BaseEnsemble
from dlecosys.shared.ensembling.pruning import PruningResult, select_estimators
from dlecosys.shared.run import EnsembleLayout
from dlecosys.shared.training import run_training


def _remap_categorical_cols(categorical_cols: list[int], feature_idx: torch.Tensor) -> list[int]:
    """Translate cat col indices from full feature space to positions inside a feature subset.

    Cat cols whose index is not in the subset are dropped (estimator simply doesn't see them).
    """
    feature_list = feature_idx.tolist()
    pos = {c: i for i, c in enumerate(feature_list)}
    return [pos[c] for c in categorical_cols if c in pos]

__all__ = ["run_ensemble"]

logger = logging.getLogger(__name__)


def _load_pool(layout: EnsembleLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the raw train pool (preprocess writes the full pool here in ensemble mode)."""
    pool = torch.load(layout.data_path("train"), weights_only=True)
    return pool["X"], pool["y"]


def _clear_console() -> None:
    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


def _render_header(
    run_name: str,
    estimator_id: int,
    n_estimators: int,
    monitor: str,
    bootstrap_size: int,
    oob_size: int,
    feature_count: int,
    n_features_total: int,
    seed: int,
    best: Dict | None,
    completed: int,
) -> None:
    _clear_console()
    print("=" * 72)
    print(f"Ensemble: {run_name}  |  Estimator {estimator_id + 1}/{n_estimators}")
    print("-" * 72)
    print(f"Metric     : {monitor} (minimize)")
    if best is not None:
        print(f"Best so far: {best['value']:.5f} (estimator {best['id']})")
    else:
        print("Best so far: — (no estimators completed yet)")
    print(f"Completed  : {completed}/{n_estimators}")
    print("-" * 72)
    print("This estimator:")
    print(f"    bootstrap size : {bootstrap_size}")
    print(f"    OOB size       : {oob_size}")
    print(f"    features       : {feature_count}/{n_features_total}")
    print(f"    seed           : {seed}")
    print("=" * 72)
    sys.stdout.flush()


def _print_estimator_result(estimator_id: int, oob_value: float | None, best: Dict | None) -> None:
    val_str = f"{oob_value:.5f}" if oob_value is not None else "—"
    line = f"Estimator {estimator_id} done — OOB val: {val_str}"
    if best is not None:
        line += f" | best: {best['value']:.5f} (estimator {best['id']})"
    print(line)


def _render_leaderboard(
    run_name: str,
    monitor: str,
    aggregation: str,
    n_estimators: int,
    results: List[Dict],
    best: Dict | None,
    pruning: PruningResult,
    top_k: int = 10,
) -> None:
    _clear_console()
    completed = [r for r in results if r["oob_value"] is not None]
    ranked = sorted(completed, key=lambda r: r["oob_value"])
    selected = set(pruning.selected)

    print("=" * 72)
    print(f"Ensemble complete: {run_name}")
    print("-" * 72)
    print(f"Metric       : {monitor} ({pruning.direction})")
    print(f"Aggregation  : {aggregation}")
    print(f"Completed    : {len(completed)}/{n_estimators}")
    print(
        f"Pruning      : {pruning.strategy}  "
        f"→ {len(pruning.selected)} of {pruning.n_total} estimators retained"
    )
    if best is not None:
        print(f"Best         : {best['value']:.5f} (estimator {best['id']})")
    if len(completed) > 1:
        values = [r["oob_value"] for r in completed]
        mean_v = sum(values) / len(values)
        var = sum((v - mean_v) ** 2 for v in values) / len(values)
        std_v = var ** 0.5
        print(f"OOB summary  : mean {mean_v:.5f} ± {std_v:.5f}")
    print("-" * 72)
    print(f"Top {min(top_k, len(ranked))} estimators (by OOB {monitor}):")
    print(f"  {'':<3}{'rank':<6}{'id':<6}{'OOB':<14}{'bootstrap':<12}{'oob_size':<10}{'features':<10}")
    for rank, r in enumerate(ranked[:top_k], start=1):
        mark = "✓" if r["estimator_id"] in selected else " "
        print(
            f"  {mark:<3}{rank:<6}{r['estimator_id']:<6}"
            f"{r['oob_value']:<14.5f}{r['bootstrap_size']:<12}{r['oob_size']:<10}"
            f"{r.get('feature_count', '—'):<10}"
        )
    if pruning.strategy != "none":
        print(f"(✓ = retained by pruning; used at inference)")
    print("=" * 72)
    sys.stdout.flush()


def run_ensemble(cfg: PipelineConfig, layout: EnsembleLayout) -> List[Dict]:
    """
    Train the full ensemble defined by ``cfg.ensemble``. Returns a list of
    per-estimator summaries (id, final OOB value, bootstrap/OOB sizes).
    """
    if cfg.ensemble is None:
        raise ValueError("run_ensemble requires a config with a non-null ensemble section")
    if cfg.ensemble.distributed.enabled:
        raise NotImplementedError(
            "Per-estimator DDP is not implemented in v1. Set "
            "ensemble.distributed.enabled: false. See TODO.md."
        )

    from dlecosys.shared.ensembling.build import build_ensemble
    ensemble = build_ensemble(cfg.ensemble)

    pool_X, pool_y = _load_pool(layout)
    pool_size = len(pool_X)
    n_features_total = pool_X.shape[1]
    monitor = cfg.training.callbacks.early_stopping.monitor

    logger.info(
        "ensemble '%s' — %d estimators, pool size %d, aggregation %s",
        cfg.experiment.name,
        cfg.ensemble.n_estimators,
        pool_size,
        cfg.ensemble.aggregation,
    )

    results: List[Dict] = []
    best: Dict | None = None

    for estimator_id in range(cfg.ensemble.n_estimators):
        rng = ensemble.rng_for(estimator_id)
        bootstrap_idx, oob_idx = ensemble.generate_bootstrap(pool_size, estimator_id, rng)
        feature_idx = ensemble.generate_feature_subset(n_features_total, estimator_id, rng)

        _render_header(
            run_name=cfg.experiment.name,
            estimator_id=estimator_id,
            n_estimators=cfg.ensemble.n_estimators,
            monitor=monitor,
            bootstrap_size=int(bootstrap_idx.numel()),
            oob_size=int(oob_idx.numel()),
            feature_count=int(feature_idx.numel()),
            n_features_total=n_features_total,
            seed=ensemble.seed + estimator_id,
            best=best,
            completed=len(results),
        )

        # Raw bootstrap + OOB, then slice feature columns
        X_boot = pool_X[bootstrap_idx][:, feature_idx]
        y_boot = pool_y[bootstrap_idx]
        has_oob = oob_idx.numel() > 0
        X_oob = pool_X[oob_idx][:, feature_idx] if has_oob else None
        y_oob = pool_y[oob_idx] if has_oob else None

        # Remap categorical column indices from full feature space to subset positions
        sub_cat_cols = _remap_categorical_cols(cfg.data.categorical_cols, feature_idx)

        # Fit ephemeral transforms on bootstrap subset, persist for later inference
        transforms = fit_transforms(
            X_boot,
            categorical_cols=sub_cat_cols,
            scaling=cfg.data.scaling,
        )
        X_boot_t = apply_transforms(X_boot, transforms)
        X_oob_t = apply_transforms(X_oob, transforms) if has_oob else None

        est_layout = layout.estimator_layout(estimator_id)
        est_layout.create(overwrite=True)

        if transforms.scaler is not None:
            transforms.scaler.save(str(layout.estimator_scaler_path(estimator_id)))
        if transforms.tokenizers:
            layout.estimator_tokenizers_dir(estimator_id).mkdir(parents=True, exist_ok=True)
            for col_idx, tok in transforms.tokenizers.items():
                tok.save(str(layout.estimator_tokenizers_dir(estimator_id) / f"tokenizer_col{col_idx}.pt"))

        torch.save(bootstrap_idx, layout.estimator_sample_indices_path(estimator_id))
        torch.save(feature_idx, layout.estimator_feature_indices_path(estimator_id))

        # Build a per-estimator config with input_dim matching this estimator's feature subset
        est_cfg_dict = cfg.model_dump()
        est_cfg_dict["model"]["params"]["input_dim"] = int(feature_idx.numel())
        est_cfg = PipelineConfig(**est_cfg_dict)

        # Train via the standard lane — EarlyStopping targets OOB loss when present
        val_ds = TensorDataset(X_oob_t, y_oob) if has_oob else None
        trainer = run_training(
            est_cfg,
            est_layout,
            train_dataset=TensorDataset(X_boot_t, y_boot),
            val_dataset=val_ds,
            summary_extras={
                "estimator_id": estimator_id,
                "oob_size": int(oob_idx.numel()),
                "feature_count": int(feature_idx.numel()),
            },
            render_summary=False,
        )

        final_logs = trainer.logger.last_log()
        oob_value_raw = final_logs.get(monitor)
        oob_value = float(oob_value_raw) if oob_value_raw is not None else None
        if oob_value is not None and not math.isfinite(oob_value):
            oob_value = None

        results.append({
            "estimator_id": estimator_id,
            "oob_value": oob_value,
            "bootstrap_size": int(bootstrap_idx.numel()),
            "oob_size": int(oob_idx.numel()),
            "feature_count": int(feature_idx.numel()),
        })

        if oob_value is not None and (best is None or oob_value < best["value"]):
            best = {"id": estimator_id, "value": oob_value}

        _print_estimator_result(estimator_id, oob_value, best)

    # Pruning selection — always runs; when disabled, all valid estimators are retained.
    pruning = select_estimators(cfg.ensemble.pruning, results)
    with open(layout.selected_path, "w") as f:
        json.dump(
            {
                "strategy": pruning.strategy,
                "direction": pruning.direction,
                "metric": monitor,
                "keep": cfg.ensemble.pruning.keep if cfg.ensemble.pruning.enabled else None,
                "n_selected": len(pruning.selected),
                "n_total": pruning.n_total,
                "selected": pruning.selected,
            },
            f,
            indent=2,
        )
    logger.info(
        "pruning: %s → %d/%d estimators retained → %s",
        pruning.strategy,
        len(pruning.selected),
        pruning.n_total,
        layout.selected_path,
    )

    # Persist ensemble summary
    summary_path = layout.ensemble_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_name": cfg.experiment.name,
                "n_estimators": cfg.ensemble.n_estimators,
                "monitor": monitor,
                "aggregation": cfg.ensemble.aggregation,
                "best": best,
                "estimators": results,
            },
            f,
            indent=2,
        )
    logger.info("ensemble summary → %s", summary_path)

    _render_leaderboard(
        run_name=cfg.experiment.name,
        monitor=monitor,
        aggregation=cfg.ensemble.aggregation,
        n_estimators=cfg.ensemble.n_estimators,
        results=results,
        best=best,
        pruning=pruning,
    )

    return results
