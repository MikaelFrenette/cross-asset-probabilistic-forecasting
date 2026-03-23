# CLI Cheat Sheet

Public entrypoints live under `scripts/`.

## Main Demo

Run the full demo stack:

```bash
bash scripts/run_demo.sh
```

## Extraction

Training extraction:

```bash
python scripts/extract_features.py --config configs/extract/train_yahoo.yaml
```

Validation extraction:

```bash
python scripts/extract_features.py --config configs/extract/val_yahoo.yaml
```

Prediction / test extraction:

```bash
python scripts/extract_features.py --config configs/extract/predict_yahoo.yaml
```

## Model Training And Prediction

Train the cross-asset attention model:

```bash
python scripts/train.py --config configs/train/main.yaml
```

Resume training from the best saved checkpoint:

```bash
python scripts/train.py \
  --config configs/train/main.yaml \
  --resume-from artifacts/demo/checkpoints/best_model.pt
```

Generate rolling model predictions:

```bash
python scripts/predict.py --config configs/predict/main.yaml
```

## GARCH Benchmark

Fit the GARCH benchmark:

```bash
python scripts/fit_garch.py --config configs/benchmark/garch.yaml
```

Generate rolling GARCH predictions:

```bash
python scripts/predict_garch.py \
  --config configs/benchmark/garch.yaml \
  --dataset data/predict_features.csv
```

## Plots

Generate the forecast plots:

```bash
python scripts/generate_plot.py \
  --manifest-path artifacts/demo/training_manifest.json \
  --predictions-csv artifacts/demo/predictions.csv \
  --dataset-csv data/predict_features.csv \
  --output-dir artifacts/demo/plots
```

## Metrics

Generate the metrics report:

```bash
python scripts/generate_metrics_report.py \
  --train-csv data/train_features.csv \
  --validation-csv data/val_features.csv \
  --dataset-csv data/predict_features.csv \
  --target-column return \
  --model cross_asset=artifacts/demo/predictions.csv \
  --model garch=artifacts/demo/garch/garch_predictions.csv \
  --output-dir artifacts/demo/metrics
```
