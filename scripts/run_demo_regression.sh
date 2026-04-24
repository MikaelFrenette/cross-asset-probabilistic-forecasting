#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/demo/regression.yaml"
OVERWRITE="${1:-}"

FLAGS=""
if [ "$OVERWRITE" = "--overwrite" ]; then
  FLAGS="--overwrite"
fi

echo "=== Demo: Regression ==="
python scripts/generate_synthetic.py --task regression
python scripts/preprocess.py --config "$CONFIG" $FLAGS
python scripts/train.py      --config "$CONFIG" $FLAGS
python scripts/predict.py    --config "$CONFIG" $FLAGS
echo "=== Done ==="
