# CLI Cheat Sheet

Every command in `dlecosys` takes `--config <path>`. The config drives everything — training, tuning, ensembling, inference all read from the same YAML.

---

## The five scripts

| Script | What it does | When to run |
|---|---|---|
| `scripts/generate_synthetic.py` | Writes a demo CSV to `data/synthetic/` | Once, if you want to play with the demo configs |
| `scripts/preprocess.py`         | CSV/Parquet → validated, scaled, split tensors | Before any train/tune/ensemble |
| `scripts/train.py`              | Trains one model (single-GPU or DDP) | Standard training lane |
| `scripts/tune.py`               | Runs a hyperparameter search | When you want to find good hyperparams |
| `scripts/ensemble.py`           | Trains N bootstrap estimators | When you want a bagging ensemble |
| `scripts/predict.py`            | Writes test predictions to `predictions.csv` | After train or ensemble |
| `scripts/evaluate.py`           | Computes metrics from `predictions.csv` → `evaluation.json` | After predict |

---

## Standard single-model workflow

```bash
python scripts/preprocess.py --config configs/my_project/exp.yaml
python scripts/train.py      --config configs/my_project/exp.yaml
python scripts/predict.py    --config configs/my_project/exp.yaml
python scripts/evaluate.py   --config configs/my_project/exp.yaml
```

Outputs land at `outputs/<experiment.name>/`.

---

## Multi-GPU (DDP)

Set in the config:
```yaml
distributed:
  enabled: true
  backend: nccl
```

Launch with `torchrun`:
```bash
torchrun --nproc_per_node=N scripts/train.py --config <cfg>
```

Everything else (preprocess, predict, evaluate) stays single-process.

---

## Hyperparameter tuning

Add a `tuning:` block to your config:
```yaml
tuning:
  study_name: my_study
  sampler: grid                  # grid | random
  splitter:
    type: kfold                  # holdout | kfold | stratified_kfold
    n_splits: 5
  search_space:
    training.optimizer.lr: [1.0e-4, 1.0e-3, 1.0e-2]
    model.params.dropout: [0.0, 0.2, 0.4]
```

Then:
```bash
python scripts/preprocess.py --config <tuning_cfg>
python scripts/tune.py        --config <tuning_cfg>
```

Tuning writes three artifacts at `outputs/<study_name>/`:
- `best_params.yaml` — just the winning hyperparameter values
- `best_config.yaml` — a ready-to-run config with the winning params applied and `tuning: null`
- `trials.csv` — every trial + its metric value

Continue with the winner:
```bash
python scripts/preprocess.py --config outputs/<study_name>/best_config.yaml
python scripts/train.py      --config outputs/<study_name>/best_config.yaml
python scripts/predict.py    --config outputs/<study_name>/best_config.yaml
python scripts/evaluate.py   --config outputs/<study_name>/best_config.yaml
```

---

## Bagging ensemble

Add an `ensemble:` block to your (typically post-tuning) config:
```yaml
ensemble:
  type: bagging
  n_estimators: 20
  aggregation: mean              # mean | median | soft_vote | majority
  sample_bootstrapper:
    type: with_replacement       # with_replacement | no_bootstrap
    max_samples: 1.0
  feature_bootstrapper:
    type: all                    # all | random_subspace
    max_features: 1.0
  pruning:
    enabled: true
    keep: 10
    strategy: top_n
```

Then:
```bash
python scripts/preprocess.py --config <ensemble_cfg>
python scripts/ensemble.py   --config <ensemble_cfg>    # OOB-as-val training per estimator
python scripts/predict.py    --config <ensemble_cfg>    # aggregates across survivors
python scripts/evaluate.py   --config <ensemble_cfg>
```

**Do not set `tuning:` and `ensemble:` in the same config.** Tune first, then ensemble the winner.

---

## TensorBoard

Enable in the config:
```yaml
training:
  callbacks:
    tensorboard:
      enabled: true
```

View logs (install `tensorboard` first — it's an optional dep):
```bash
pip install tensorboard
tensorboard --logdir outputs/<run_name>/logs/tensorboard
```

---

## Overwrite policies

Each script refuses to clobber existing artifacts by default. Pass `--overwrite` to force.

```bash
python scripts/preprocess.py --config <cfg> --overwrite
python scripts/train.py      --config <cfg> --overwrite
python scripts/predict.py    --config <cfg> --overwrite
```

---

## Tests

```bash
pytest                                          # full suite
pytest tests/shared/training/                   # one package
pytest tests/shared/data/test_splitters.py -v   # one file
pytest --tb=short -q                            # compact output
```

---

## Common gotchas

- Tuning + DDP in the same config is not supported. Tune on single GPU; DDP-train the winner.
- `preprocess.py` writes **raw** tensors (no scaling) when `tuning.scale_splits: true` OR `ensemble:` is set, because per-fold / per-estimator scaling happens on the fly.
- Config inheritance is capped at one level. A child's `base:` must point to a config with `base: null`.
- Every custom model must be imported in `src/dlecosys/models/__init__.py` so the `@register` decorator fires.

See [AGENTS.md](../AGENTS.md) for the "task → edit site" map when adding new components.
