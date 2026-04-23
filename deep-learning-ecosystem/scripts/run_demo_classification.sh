#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/demo/classification.yaml"
OVERWRITE="${1:-}"

FLAGS=""
if [ "$OVERWRITE" = "--overwrite" ]; then
  FLAGS="--overwrite"
fi

echo "=== Demo: Classification ==="
python scripts/generate_synthetic.py --task classification
python scripts/preprocess.py --config "$CONFIG" $FLAGS
python scripts/train.py      --config "$CONFIG" $FLAGS
python scripts/predict.py    --config "$CONFIG" $FLAGS
echo "=== Done ==="
