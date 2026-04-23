"""
predict.py — Run inference on the test split for a pipeline run.

Single-model mode (no ``ensemble`` section)
-------------------------------------------
Loads the trained checkpoint, applies the persisted scaler/tokenizers to the
test set, writes ``predictions.csv``.

Ensemble mode (``ensemble`` section present)
--------------------------------------------
Loads every estimator's checkpoint + per-estimator scaler/tokenizers, applies
each estimator's transforms to the raw test tensor, collects predictions, and
aggregates via ``cfg.ensemble.aggregation`` (mean / median / soft_vote / majority).

Usage
-----
    python scripts/predict.py --config <cfg>
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dlecosys.models  # noqa: F401 — registers bundled models with the factory

from dlecosys.shared.config import configure_logging, load_config
from dlecosys.shared.config.schema import PipelineConfig
from dlecosys.shared.ensembling.aggregation import aggregate
from dlecosys.shared.inference import Predictor
from dlecosys.shared.models import ModelFactory
from dlecosys.shared.preprocessing import IntegerTokenizer, MinMaxScaler, StandardScaler
from dlecosys.shared.run import EnsembleLayout, RunLayout

logger = logging.getLogger(__name__)

_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


def _transform_X(
    X,
    cfg,
    scaler_path: Path | None,
    tokenizer_path_fn,
    cat_cols: list[int] | None = None,
):
    """Apply tokenizers + scaler (if present) to X. Mutates and returns X.

    ``cat_cols`` defaults to ``cfg.data.categorical_cols`` but can be remapped
    by the caller (e.g. ensemble prediction with feature-subsampled estimators).
    """
    if cat_cols is None:
        cat_cols = cfg.data.categorical_cols
    n_features = X.shape[1]
    cont_cols = [i for i in range(n_features) if i not in cat_cols]

    for col_idx in cat_cols:
        tok = IntegerTokenizer.load(str(tokenizer_path_fn(col_idx)))
        col = tok.transform(X[:, col_idx].long().numpy())
        X[:, col_idx] = torch.from_numpy(col).float()

    if cfg.data.scaling is not None and cont_cols and scaler_path is not None and scaler_path.exists():
        scaler = _SCALERS[cfg.data.scaling].load(str(scaler_path))
        X[:, cont_cols] = scaler.transform(X[:, cont_cols]).float()

    return X


def _predict_single(cfg, args) -> None:
    layout = RunLayout(cfg.experiment.output_dir, cfg.experiment.name)

    if not layout.checkpoint_path.exists():
        logger.error("checkpoint not found at %s — run train.py first", layout.checkpoint_path)
        sys.exit(1)
    if layout.predictions_csv.exists() and not args.overwrite:
        logger.error("predictions already exist at %s — use --overwrite to redo", layout.predictions_csv)
        sys.exit(1)

    # test.pt was already transformed by preprocess.py in the single-model lane
    # (scaler + tokenizers applied + persisted as artifacts). We load it as-is.
    test_data = torch.load(layout.data_path("test"), weights_only=True)
    X_test, y_test = test_data["X"], test_data["y"]

    model = ModelFactory.build(cfg.model.name, cfg.model.params)
    predictor = Predictor.from_checkpoint(str(layout.checkpoint_path), model)
    preds = predictor.predict(X_test, batch_size=cfg.inference.batch_size)

    y_true = y_test.flatten().tolist()
    y_pred = preds.flatten().tolist()

    with open(layout.predictions_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        writer.writerows(zip(y_true, y_pred))

    logger.info("predictions written to %s", layout.predictions_csv)


def _predict_ensemble(cfg, args) -> None:
    layout = EnsembleLayout(cfg.experiment.output_dir, cfg.experiment.name)

    if layout.predictions_csv.exists() and not args.overwrite:
        logger.error("predictions already exist at %s — use --overwrite to redo", layout.predictions_csv)
        sys.exit(1)

    test_data = torch.load(layout.data_path("test"), weights_only=True)
    X_test_raw, y_test = test_data["X"], test_data["y"]

    n_estimators = cfg.ensemble.n_estimators

    # Filter by pruning selection if available
    import json as _json
    if layout.selected_path.exists():
        with open(layout.selected_path) as f:
            selected_ids = set(_json.load(f)["selected"])
        logger.info("using %d selected estimators from %s", len(selected_ids), layout.selected_path)
    else:
        selected_ids = set(range(n_estimators))

    all_preds = []

    for estimator_id in range(n_estimators):
        if estimator_id not in selected_ids:
            continue
        est_layout = layout.estimator_layout(estimator_id)
        if not est_layout.checkpoint_path.exists():
            logger.warning("missing checkpoint for estimator %d — skipping", estimator_id)
            continue

        # Load feature indices and slice test X to this estimator's feature subset
        feature_indices_path = layout.estimator_feature_indices_path(estimator_id)
        if feature_indices_path.exists():
            feature_idx = torch.load(feature_indices_path, weights_only=True)
        else:
            feature_idx = torch.arange(X_test_raw.shape[1])

        X_test_sub = X_test_raw.clone()[:, feature_idx]

        # Remap categorical cols from full feature space to subset positions
        feature_list = feature_idx.tolist()
        pos = {c: i for i, c in enumerate(feature_list)}
        sub_cat_cols = [pos[c] for c in cfg.data.categorical_cols if c in pos]

        X_test_sub = _transform_X(
            X_test_sub,
            cfg,
            scaler_path=layout.estimator_scaler_path(estimator_id),
            tokenizer_path_fn=lambda col, eid=estimator_id: layout.estimator_tokenizers_dir(eid) / f"tokenizer_col{col}.pt",
            cat_cols=sub_cat_cols,
        )

        # Build model with this estimator's input_dim = feature subset size
        est_params = dict(cfg.model.params)
        est_params["input_dim"] = int(feature_idx.numel())
        model = ModelFactory.build(cfg.model.name, est_params)
        predictor = Predictor.from_checkpoint(str(est_layout.checkpoint_path), model)
        preds = predictor.predict(X_test_sub, batch_size=cfg.inference.batch_size)
        all_preds.append(preds)
        logger.info("estimator %d predicted", estimator_id)

    if not all_preds:
        logger.error("no estimator checkpoints found under %s", layout.ensemble_dir)
        sys.exit(1)

    stacked = torch.stack(all_preds, dim=0)  # (n_est, n_samples, out_dim)
    aggregated = aggregate(stacked, mode=cfg.ensemble.aggregation)

    y_true = y_test.flatten().tolist()
    y_pred = aggregated.flatten().tolist() if aggregated.dim() > 1 else aggregated.tolist()

    with open(layout.predictions_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        writer.writerows(zip(y_true, y_pred))

    logger.info(
        "ensemble predictions written to %s (aggregation: %s, n=%d estimators)",
        layout.predictions_csv,
        cfg.ensemble.aggregation,
        len(all_preds),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on the test split.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)

    if cfg.ensemble is not None:
        _predict_ensemble(cfg, args)
    else:
        _predict_single(cfg, args)


if __name__ == "__main__":
    main()
