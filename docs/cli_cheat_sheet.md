# CLI Cheat Sheet

Every command takes `--config <path>`. The YAML drives everything — data,
features, model, training, inference, ensembling, tuning, benchmarks,
plotting, metrics.

---

## The scripts

| Script | What it does | When to run |
|---|---|---|
| `scripts/preprocess.py` | Raw CSV → fitted `PreprocessingBundle` + vectorized `(B, I, T, K)` panels on disk | Before `train.py` (single-model lane). The `tune.py` and `ensemble.py` lanes re-vectorize internally and do NOT need this. |
| `scripts/train.py`      | Trains one panel model (single-GPU or DDP) → `model.pt` + `training_manifest.json` | Standard training |
| `scripts/tune.py`       | Optuna grid / random search over a time-series CV splitter → `best_config.yaml` + `trials.csv` | When you want to find good hyperparameters |
| `scripts/ensemble.py`   | Trains N bootstrap estimators with OOB-as-val + top-N pruning → `ensemble/estimator_k/` + `ensemble_manifest.json` | When you want a bagging ensemble |
| `scripts/benchmark.py`  | Fits AR(1) / UnconditionalMean / GARCH(1,1) → `predictions.csv` in the same schema as `predict.py` | When you need baseline comparisons |
| `scripts/predict.py`    | Rolling Gaussian inference on an external CSV → `predictions.csv`. Auto-detects single-model vs ensemble manifest. | After training or ensembling |
| `scripts/plot.py`       | Per-asset three-panel forecast report (μ, σ, μ/σ signal) → `plots/<asset>.png` | After `predict.py` |
| `scripts/plot_calibration.py` | Two-panel σ showcase — σ vs rolling realized vol for a focus asset + panel-wide reliability diagram → `sigma_calibration.png` | After `predict.py` |
| `scripts/plot_signal.py` | Two-panel signal showcase — per-bucket annualized realized return + Q5−Q1 cumulative spread vs benchmark → `signal_decile.png` | After `predict.py` |
| `scripts/metrics.py`    | Joins predictions + realized targets; emits NLL / CRPS / MAE / RMSE / bias / coverage / mean σ; optional skill scores vs a reference | After `predict.py` (and optional benchmarks) |

---

## Standard single-model workflow

```bash
python scripts/preprocess.py --config configs/density_model/yahoo_volatility.yaml
python scripts/train.py      --config configs/density_model/yahoo_volatility.yaml
python scripts/predict.py    --config configs/density_model/yahoo_predict.yaml
python scripts/plot.py       --config configs/density_model/yahoo_predict.yaml
python scripts/metrics.py    --config configs/density_model/yahoo_predict.yaml
```

Outputs land at `outputs/<experiment_name>/` (set by `artifacts.output_dir`):
```
preprocessing.json                fitted per-asset scaler + global tokenizer
train_panel.pt                    vectorized train (B, I, T, K)
val_panel.pt                      vectorized val
model.pt                          trained weights
training_manifest.json            everything predict.py needs to rebuild
checkpoints/best_model.pt         from ModelCheckpoint (if enabled)
predictions.csv                   rolling forecasts (asset, date, mu, sigma)
plots/<asset>_report.png          per-asset 3-panel reports
metrics.{json,md}                 + metrics_per_asset.csv
```

---

## Picking the model

Two registered density architectures:

```yaml
model:
  name: "causal_cross_asset_transformer"   # tail mode: forecast after input window
features:
  target_mode: "tail"                      # must match the model
```
vs.
```yaml
model:
  name: "causal_next_step_transformer"     # GPT-style per-position supervision
features:
  target_mode: "next_step"                 # must match the model
```

The tail model emits `(B, I, out_steps, 2·target_dim)` — one prediction
after the input window. The next-step model emits `(B, I, T, 2·target_dim)`
— one prediction per input position, supervised against the input shifted
by one. Cross-asset attention applies once (tail, on the final hidden
state) vs. at every timestep (next-step).

`predict.py` reads `manifest.target_mode` and extracts the correct
forecast position (position 0 for tail, position T-1 for next-step) into
the one-row-per-(asset, forecast_date) CSV.

---

## Multi-GPU (DDP)

Set in the config:
```yaml
training:
  device: "cuda"
distributed:
  enabled: true
  backend: "nccl"
```

Launch with `torchrun`:
```bash
python scripts/preprocess.py --config configs/density_model/yahoo_volatility_ddp.yaml
torchrun --nproc_per_node=N scripts/train.py --config configs/density_model/yahoo_volatility_ddp.yaml
```

Everything else (predict, plot, metrics) stays single-process. Rank 0
writes the artifacts; other ranks participate in gradient sync only.

---

## Hyperparameter tuning

Add a `tuning:` block to the config (see
[configs/density_model/yahoo_volatility_tuning_bs_lr.yaml](../configs/density_model/yahoo_volatility_tuning_bs_lr.yaml)):
```yaml
tuning:
  study_name: "my_study_v1"
  direction: "minimize"
  metric: "val_loss"
  sampler: "grid"                # or "random"
  epochs_per_fold: 20
  pruner:
    enabled: true
    type: "median"
    n_warmup_steps: 1            # skip the first epoch before pruning activates
    n_startup_trials: 3          # complete 3 trials fully before comparing
  splitter:
    type: "expanding_window"     # holdout | expanding_window | walk_forward
    n_splits: 3
  search_space:
    training.batch_size: [16, 32, 64]
    training.learning_rate: [1.0e-4, 3.0e-4, 1.0e-3]
  post_tuning:                   # deep-merged into the emitted best_config.yaml
    training:
      max_epochs: 50             # the winner trains at full budget
    early_stopping:
      enabled: true
```

Then:
```bash
python scripts/tune.py --config configs/density_model/yahoo_volatility_tuning_bs_lr.yaml
```

Tuning writes three artifacts at `outputs/<output_dir>/studies/<study_name>/`:
- `best_params.yaml` — winning hyperparameter values + best objective value
- `best_config.yaml` — full config with the winner applied, `tuning: null`, and `post_tuning` merged
- `trials.csv` — every trial + state (COMPLETE / PRUNED / FAIL) + params

Continue with the winner:
```bash
python scripts/preprocess.py --config outputs/<output_dir>/studies/<study_name>/best_config.yaml --overwrite
python scripts/train.py      --config outputs/<output_dir>/studies/<study_name>/best_config.yaml
python scripts/predict.py    --config configs/density_model/yahoo_predict.yaml
python scripts/plot.py       --config configs/density_model/yahoo_predict.yaml
python scripts/metrics.py    --config configs/density_model/yahoo_predict.yaml
```

Per-fold clear + TF-Tuner-style display shows best-so-far hyperparameters,
progress counters, running fold losses, and a final leaderboard ranking
all trials by the metric (with pruning counts).

---

## Bagging ensemble

Add an `ensemble:` block (typically to a post-tuning best_config, see
[configs/density_model/yahoo_volatility_ensemble_real.yaml](../configs/density_model/yahoo_volatility_ensemble_real.yaml)):
```yaml
ensemble:
  type: "bagging"
  n_estimators: 15
  aggregation: "mean"            # mean | median (for μ; σ uses law of total variance)
  sample_bootstrapper:
    type: "with_replacement"     # with_replacement | no_bootstrap
    max_samples: 0.8             # fraction of unique forecast dates per estimator
  feature_bootstrapper:
    type: "all"                  # all | random_subspace
    max_features: 1.0
  pruning:
    enabled: true
    keep: 10                     # top-N by OOB val_loss
    strategy: "top_n"
  seed: 42
```

Then:
```bash
python scripts/ensemble.py --config configs/density_model/yahoo_volatility_ensemble_real.yaml
python scripts/predict.py  --config configs/density_model/yahoo_predict_ensemble_real.yaml   # manifest_path points at ensemble_manifest.json
python scripts/plot.py     --config configs/density_model/yahoo_predict_ensemble_real.yaml
python scripts/metrics.py  --config configs/density_model/yahoo_predict_ensemble_real.yaml
```

The ensemble runner fits a global tokenizer once, vectorizes the raw
panel once, then for each estimator: samples in-bag forecast dates,
fits a per-asset scaler on that slice, slices tensors along B and I
axes, applies the scaler at tensor level (broadcast subtract/divide),
and trains with OOB-as-val. Top-N pruning keeps the best survivors by
OOB loss.

Do NOT set both `tuning:` and `ensemble:` in the same config — tune
first, then add the ensemble block to the emitted best_config.

---

## Benchmarks

Three classical baselines — configs live in
[configs/density_model/benchmarks/](../configs/density_model/benchmarks/):

| Benchmark | Type | Notes |
|---|---|---|
| Unconditional mean | Point only | Per-asset historical mean over the training CSV; typically used as the RMSE skill-score reference |
| AR(1) | Point only | Per-asset closed-form OLS fit; one-step-ahead rolling forecast |
| GARCH(1,1) | Full density (μ + σ) | `arch` package; Constant-mean + GARCH(1,1); variance rolled forward one step at a time through the test window |

Run each independently:
```bash
python scripts/benchmark.py --config configs/density_model/benchmarks/unconditional_mean.yaml
python scripts/benchmark.py --config configs/density_model/benchmarks/ar1.yaml
python scripts/benchmark.py --config configs/density_model/benchmarks/garch.yaml
```

Each writes `predictions.csv` in the same four-column schema used by
`scripts/predict.py`. Point-only benchmarks leave `sigma = NaN`, which
signals the metrics script to skip density metrics (NLL, CRPS, coverage)
for those rows.

---

## Metrics + benchmark comparison

Basic — primary model only:
```bash
python scripts/metrics.py --config configs/density_model/yahoo_predict.yaml
```

With benchmarks + RMSE skill score against the unconditional-mean
reference:
```bash
python scripts/metrics.py \
    --config configs/density_model/yahoo_predict.yaml \
    --benchmarks outputs/benchmarks/unconditional_mean/predictions.csv \
                 outputs/benchmarks/ar1/predictions.csv \
                 outputs/benchmarks/garch/predictions.csv \
    --reference  outputs/benchmarks/unconditional_mean/predictions.csv
```

Metrics computed:

- `mean_nll` — training-convention Gaussian NLL: `mean((y − μ)² / σ² + log σ²)`
- `crps` — Gaussian CRPS (closed form): lower is better; ~ σ/√π for calibrated forecasts
- `mae_mu`, `rmse_mu`, `bias_mu` — point-forecast quality (bias = E[μ − target])
- `rmse_skill` — `rmse_model / rmse_reference` (overall scalar + per-asset column)
- `coverage_1s`, `coverage_2s` — empirical P(|y − μ| ≤ σ) vs targets 0.683 / 0.955
- `sigma_mean` — sharpness

Output files land alongside the primary `predictions.csv`:
- `metrics.json` — overall + per-asset payload for every model
- `metrics_per_asset.csv` — long format: one row per (model, asset)
- `metrics.md` — markdown version of the stdout report

---

## Overwrite policies

`preprocess.py` refuses to clobber existing artifacts. Pass `--overwrite`:
```bash
python scripts/preprocess.py --config <cfg> --overwrite
```

`train.py`, `tune.py`, `ensemble.py`, `predict.py`, `plot.py`,
`plot_calibration.py`, `plot_signal.py`, `metrics.py`, `benchmark.py`
overwrite by default (they write to deterministic paths derived from
the config).

---

## Tests

```bash
pytest                                     # full suite
pytest tests/shared/training/              # one package
pytest tests/shared/data/test_splitters.py -v   # one file
pytest --tb=short -q                       # compact output
```

---

## Common gotchas

- **`features.target_mode`** must match the chosen model. `"tail"` pairs
  with `causal_cross_asset_transformer` / `temporal_only_transformer`;
  `"next_step"` pairs with `causal_next_step_transformer`. Mismatched
  configs produce misaligned target/output shapes and obvious training
  failures.
- **`preprocess.py` is only for the single-model lane.** `tune.py` and
  `ensemble.py` load the raw CSVs directly and re-fit preprocessing
  per fold / per estimator (leak-free).
- **`max_epochs`** under `tuning:` is ignored by the tuning loop — it
  uses `tuning.epochs_per_fold` instead. Use `post_tuning.training.max_epochs`
  to set the full-budget value for the emitted `best_config.yaml`.
- **`ensemble:` + `tuning:` in the same config is not supported.** Tune
  first, then add `ensemble:` to the winning config.
- **Per-estimator GPU** in the ensemble lane trains serially. DDP
  per-estimator is not implemented; use DDP for the final single-model
  training of a tuned config if you need multi-GPU.
- **`distributed:` + `tuning:` in the same config is not supported.**
  Tune on single-GPU; DDP-train the winner.
