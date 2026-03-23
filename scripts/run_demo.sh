#!/usr/bin/env bash
set -euo pipefail

# End-to-end repo demo workflow.
# This script runs:
# 1. Yahoo feature extraction for train / validation / prediction datasets
# 2. Model training
# 3. Rolling prediction generation
# 4. GARCH benchmark fitting and rolling prediction
# 5. Forecast plotting
# 6. Metrics report generation
#
# Usage:
#   bash scripts/run_demo.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -ne 0 ]]; then
  echo "usage: bash scripts/run_demo.sh" >&2
  exit 1
fi

run_step() {
  local label="$1"
  shift
  printf '\n[%s]\n' "${label}"
  printf 'command: %s\n' "$*"
  "$@"
}

run_step "extract-train" python scripts/extract_features.py --config configs/extract/train_yahoo.yaml
run_step "extract-val" python scripts/extract_features.py --config configs/extract/val_yahoo.yaml
run_step "extract-predict" python scripts/extract_features.py --config configs/extract/predict_yahoo.yaml

run_step "train" python scripts/train.py --config configs/train/main.yaml
run_step "predict" python scripts/predict.py --config configs/predict/main.yaml

run_step "garch-fit" python scripts/fit_garch.py --config configs/benchmark/garch.yaml
run_step "garch-predict" \
  python scripts/predict_garch.py \
    --config configs/benchmark/garch.yaml \
    --dataset data/predict_features.csv

run_step "forecast-plot" \
  python scripts/generate_plot.py \
    --manifest-path artifacts/demo/training_manifest.json \
    --predictions-csv artifacts/demo/predictions.csv \
    --dataset-csv data/predict_features.csv \
    --output-dir artifacts/demo/plots

METRICS_CMD=(
  python scripts/generate_metrics_report.py
  --train-csv data/train_features.csv
  --validation-csv data/val_features.csv
  --dataset-csv data/predict_features.csv
  --target-column return
  --model cross_asset=artifacts/demo/predictions.csv
  --model garch=artifacts/demo/garch/garch_predictions.csv
  --output-dir artifacts/demo/metrics
)

run_step "metrics" "${METRICS_CMD[@]}"

printf '\n[done]\n'
printf 'predictions: artifacts/demo/predictions.csv\n'
printf 'garch predictions: artifacts/demo/garch/garch_predictions.csv\n'
printf 'plots: artifacts/demo/plots\n'
printf 'metrics: artifacts/demo/metrics\n'
printf 'garch artifacts: artifacts/demo/garch\n'
