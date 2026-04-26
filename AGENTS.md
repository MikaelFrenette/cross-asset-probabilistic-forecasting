# AGENTS.md

Entry point for a coding agent dropped into this repository. Read it first —
it maps every extension point a user will want to reach.

Companion docs:
- [README.md](README.md) — what the repo is and why it exists
- [CONVENTION.md](CONVENTION.md) — hard coding rules (non-negotiable)
- [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md) — every command at a glance
- [CLAUDE.md](CLAUDE.md) — stack, commands, naming
- [TODO.md](TODO.md) — deferred cleanups, known gaps
- [IDEA.md](IDEA.md) — architectural ideas, alternatives

---

## What this repo does

A panel density-forecasting library. Train a causal transformer to emit a
Gaussian predictive distribution `(μ, σ)` per asset per forecast date, then
operate it through every production lane:

- **Preprocess:** long-format CSV → fitted per-asset scaler + tokenizer + vectorized `(B, I, T, K)` panels on disk
- **Train:** any registered panel model, single-GPU or DDP (via `torchrun`)
- **Tune:** Optuna grid/random over a panel time-series CV splitter; emits a ready-to-run `best_config.yaml`
- **Ensemble:** bagging over forecast dates with OOB-as-val and top-N pruning
- **Benchmark:** AR(1), unconditional mean, GARCH(1,1) — same `predictions.csv` schema
- **Predict / Plot / Metrics:** rolling Gaussian forecasts, per-asset 3-panel reports, density-aware metrics with skill scores
- **Portfolio (illustrative):** an example downstream consumer of `predictions.csv` — not a maintained lane

Configuration is YAML, validated by Pydantic v2. A single config object flows
through every script; sibling configs (`PanelPredictConfig`,
`PanelBenchmarkConfig`, `PanelPortfolioConfig`) drive the standalone lanes.

---

## Decision tree — which lane to pick

```
User has labelled panel data (long-format CSV with date, asset_id, return)
    │
    ├── wants one model                  → preprocess → train → predict → plot → metrics
    ├── wants to search hyperparameters  → preprocess (tuning mode) → tune → preprocess → train (best_config) → predict → ...
    ├── wants a bagging ensemble         → preprocess → ensemble → predict → ...
    ├── wants a baseline                 → benchmark → predict (csv already written) → metrics
    └── wants a downstream backtest      → predict → construct_portfolio → plot_portfolio
```

See [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md) for the exact commands.

---

## Repo map (scan once, keep in mind)

```
configs/
  density_model/
    yahoo_volatility*.yaml             single-model + DDP + smoke configs
    yahoo_volatility_tuning*.yaml      Optuna search configs
    yahoo_volatility_ensemble*.yaml    bagging configs
    yahoo_predict*.yaml                inference configs (matching names)
    benchmarks/                        AR(1), UnconditionalMean, GARCH(1,1)
    portfolio/baseline.yaml            downstream example config
data/                                  train/val/predict feature CSVs
docs/
  cli_cheat_sheet.md                   full command reference
  images/                              README assets
src/density_model/
  models/                              <-- USER ADDS PANEL MODELS HERE
    panel_base.py                      abstract BasePanelModel
    cross_asset.py                     causal_cross_asset_transformer (target_mode=tail)
    causal_next_step.py                causal_next_step_transformer (target_mode=next_step)
    temporal_only.py                   temporal_only_transformer
    __init__.py                        imports each module so @register fires
  portfolio/                           illustrative downstream example
    config.py                          PanelPortfolioConfig
    alpha/                             alpha builder dispatch (mu_sigma_zscore)
    risk_model/                        covariance estimator dispatch (rolling)
    optimizer/                         optimizer dispatch (mean_variance closed-form / SLSQP)
    backtest.py                        strict-no-lookahead daily loop
    metrics.py                         Sharpe / annual return / vol / MDD / hit / leverage / turnover / Eff-N
    plotting.py                        cumulative return vs benchmark
    run.py                             end-to-end runner
  shared/                              library code
    config/
      panel_schema.py                  every Pydantic config in the panel pipeline
      logging.py                       configure_logging()
      trainer_config.py                trainer dataclass derived from PanelTrainingConfig
    data/panel/                        vectorizer, dataset, splitter, schema, calendar, sources
    preprocessing/panel/               per-asset scaler + tokenizer + bundle
    models/                            ModelFactory + @register decorator + base ModelConfig
    benchmarks/                        AR(1), UnconditionalMean, GARCH(1,1) + run.py + build.py
    training/                          BaseTrainer, callbacks, panel_trainer, panel_distributed, panel_losses, panel_run, panel_preprocess, panel_manifest
    tuning/panel/                      Optuna objective, study, splitter_builder, pruning_callback, best_config emitter
    ensembling/panel/                  bagging runner, bootstrappers, aggregation, pruning, manifest, display
    inference/                         panel_predictor (auto single-vs-ensemble dispatch)
    evaluation/                        panel_plot, panel_metrics
scripts/                               thin entrypoints — config → library → exit
  preprocess.py                        raw CSVs → fitted bundle + vectorized panels
  train.py                             single-GPU or DDP via torchrun
  tune.py                              Optuna study; emits best_config.yaml
  ensemble.py                          N bootstrap estimators, OOB-as-val
  benchmark.py                         AR(1) / UnconditionalMean / GARCH(1,1)
  predict.py                           rolling Gaussian inference (auto single/ensemble)
  plot.py                              per-asset 3-panel forecast report
  metrics.py                           NLL / CRPS / MAE / RMSE / coverage + optional skill score
  construct_portfolio.py               downstream example (not a maintained lane)
  plot_portfolio.py                    cumulative return vs SPY
tests/                                 pytest suite (currently empty — see TODO.md)
outputs/                               per-run artifacts (gitignored)
```

---

## "I want to..." task → edit sites

| Task | Edit | Pattern to copy |
|---|---|---|
| Add a panel model | `src/density_model/models/<name>.py` + import in `models/__init__.py` | [cross_asset.py](src/density_model/models/cross_asset.py) — `@register("name")` + `ModelConfig` subclass + `from_config` + `BasePanelModel` |
| Add a target mode | `src/density_model/shared/training/panel_losses.py` + `PanelFeaturesConfig.target_mode` regex in `panel_schema.py` + propagate in `panel_preprocess.py` | existing `tail` / `next_step` branches |
| Add a panel splitter (tuning CV) | `src/density_model/shared/data/panel/splitter.py` (subclass `BasePanelSplitter`) + `build_panel_splitter` dispatch in `tuning/panel/splitter_builder.py` + `PanelSplitterConfig.type` regex | `PanelExpandingWindowSplitter`, `PanelWalkForwardSplitter`, `PanelHoldoutSplitter` |
| Add a benchmark | `src/density_model/shared/benchmarks/<name>.py` (subclass `BaseBenchmark`) + dispatch in `benchmarks/build.py` + `PanelBenchmarkConfig.type` regex | `AR1Benchmark`, `UnconditionalMeanBenchmark`, `GarchBenchmark` |
| Add a sample bootstrapper | `src/density_model/shared/ensembling/panel/bootstrappers.py` (subclass `BaseSampleBootstrapper`) + `_build_sample_bootstrapper` dispatch in `ensembling/panel/run.py` + `PanelSampleBootstrapperConfig.type` regex | `DateBootstrapper`, `NoSampleBootstrapper` |
| Add a feature bootstrapper | same file, subclass `BaseFeatureBootstrapper` + `_build_feature_bootstrapper` dispatch + `PanelFeatureBootstrapperConfig.type` regex | `AssetSubspaceBootstrapper`, `NoFeatureBootstrapper` |
| Add an ensemble pruning strategy | `src/density_model/shared/ensembling/panel/pruning.py::select_estimators` + `PanelEnsemblePruningConfig.strategy` regex | `top_n` |
| Add an aggregation mode | `src/density_model/shared/ensembling/panel/aggregation.py` + `PanelEnsembleConfig.aggregation` regex | `mean`, `median` (for Gaussian aggregation: μ̄ + law of total variance for σ) |
| Add a callback | `src/density_model/shared/training/callbacks.py` (subclass `Callback`) + wire into `_build_callbacks` in `training/panel_run.py` if config-driven | `EarlyStopping`, `ModelCheckpoint`, `LRSchedulerCallback`, `GradNormCallback`, `TensorBoardCallback` |
| Add an optimizer / scheduler | `src/density_model/shared/config/trainer_config.py` (the trainer dataclass it builds) | search `_OPTIMIZERS` / `_SCHEDULERS` builders |
| Add an alpha builder | `src/density_model/portfolio/alpha/<name>.py` (subclass `BaseAlphaBuilder`) + `build_alpha_builder` dispatch + `PanelAlphaConfig.type` regex | `MuSigmaZScoreAlpha` |
| Add a covariance estimator | `src/density_model/portfolio/risk_model/<name>.py` + `build_covariance_estimator` dispatch + `PanelRiskModelConfig.type` regex | `RollingCovarianceEstimator` |
| Add a portfolio optimizer | `src/density_model/portfolio/optimizer/<name>.py` + `build_optimizer` dispatch + `PanelOptimizerConfig.type` regex | `MeanVarianceOptimizer` (closed-form + SLSQP fallback) |
| Add a config knob | `src/density_model/shared/config/panel_schema.py` (or `portfolio/config.py`) — add a field to the appropriate section, update the corresponding YAML configs | every existing section |

---

## Commands

All commands live in [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md).
The golden path per lane:

```bash
# Single model:  preprocess → train → predict → plot → metrics
# Tuning:        preprocess (tuning mode) → tune → preprocess → train (best_config) → predict → ...
# Ensemble:      preprocess → ensemble → predict → ...
# Benchmark:     benchmark → metrics       # benchmark.py writes predictions.csv directly
# Portfolio:     predict → construct_portfolio → plot_portfolio
```

**Recommended full workflow:** tune the architecture first (`tuning:` config →
`best_config.yaml`), then bolt on an `ensemble:` section to the winning config
and run the ensemble flow. Do not set both `tuning:` and `ensemble:` in the
same config — `preprocess.py` will reject it.

---

## Extension patterns — copy-paste templates

### Your own panel model

```python
# src/density_model/models/my_panel_model.py
from pydantic import Field
from torch import nn

from density_model.shared.models import ModelConfig, register
from density_model.models.panel_base import BasePanelModel


class MyPanelModelConfig(ModelConfig):
    n_features: int
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    dropout: float = 0.1


@register("my_panel_model")
class MyPanelModel(BasePanelModel):
    config_class = MyPanelModelConfig

    def __init__(self, n_features, n_layers, n_heads, d_model, dropout):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.mu_head = nn.Linear(d_model, 1)
        self.log_sigma_head = nn.Linear(d_model, 1)

    @classmethod
    def from_config(cls, config: MyPanelModelConfig) -> "MyPanelModel":
        return cls(
            n_features=config.n_features,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_model=config.d_model,
            dropout=config.dropout,
        )

    def forward(self, x):
        # x: (B, I, T, K)
        ...
        return mu, sigma  # both (B, I) for tail mode, (B, I, T) for next_step
```

Then in `src/density_model/models/__init__.py`:

```python
from density_model.models.my_panel_model import MyPanelModel  # noqa: F401
```

Reference in config:

```yaml
model:
  name: my_panel_model
  params:
    n_features: 1
    n_layers: 3
    d_model: 128
features:
  target_mode: tail   # or next_step
```

### Your own panel splitter

```python
# In src/density_model/shared/data/panel/splitter.py:

class PurgedKFoldSplitter(BasePanelSplitter):
    """Time-series KFold with an embargo gap to prevent leakage."""

    def __init__(self, *, n_splits: int = 5, embargo_days: int = 5):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def generate_folds(self, panel, *, date_column):
        ...
```

Then wire it:

```python
# splitter_builder.py
if cfg.type == "purged_kfold":
    return PurgedKFoldSplitter(n_splits=cfg.n_splits, embargo_days=cfg.embargo_days)

# panel_schema.py — PanelSplitterConfig.type pattern adds "purged_kfold"
```

### Your own benchmark

```python
# src/density_model/shared/benchmarks/ewma.py
from density_model.shared.benchmarks.base import BaseBenchmark


class EwmaBenchmark(BaseBenchmark):
    """EWMA volatility forecast — μ=0, σ from EWMA of squared returns."""

    def fit_predict(self, train_returns, predict_dates):
        ...
        return mu_df, sigma_df  # (date × asset)


# Then in benchmarks/build.py:
if cfg.type == "ewma":
    return EwmaBenchmark()

# panel_schema.py — PanelBenchmarkConfig.type pattern adds "ewma"
```

---

## Hard rules (do not violate)

From [CONVENTION.md](CONVENTION.md):

- **§23 No silent fallbacks.** If input is invalid, raise — never impute, clip, or coerce. Users decide how to handle failures.
- **Every module has a top docstring with an `__all__`.**
- **Config is always Pydantic-validated.** No raw dicts past the YAML boundary.
- **Scripts are thin.** Logic belongs in `src/`, scripts just wire config → library → exit.
- **Logging configured once at entrypoint** via `configure_logging(cfg.logging)`.
- **YAML extension is `.yaml`, never `.yml`.**

---

## Gotchas that trip agents

1. **Models must be imported after definition** for `@register` to fire. Add the import to `src/density_model/models/__init__.py`.
2. **`preprocess.py` is a single-model lane only.** It writes one fitted preprocessing bundle + vectorized panels. Tuning and ensemble run their own per-fold / per-estimator preprocessing internally.
3. **`max_epochs` under a `tuning:` section is ignored** for the trial loop's purposes — the trial uses the inner `training.max_epochs`. Tuning-specific budget knobs live in `tuning:`.
4. **`predict.py` auto-detects ensemble manifests.** If `outputs/<name>/ensemble_manifest.json` exists, it averages μ across estimators and combines σ via the law of total variance. Do not hand-roll aggregation.
5. **Pruning callback uses a flag, not a raise.** `CallbackList._call` swallows exceptions, so `PruningCallback` sets `should_prune` and the objective re-checks it after `trainer.train()` returns and raises `optuna.TrialPruned` from outside the swallowed scope. Do not change this without re-checking that pruned trials still appear in `trials.csv`.
6. **DDP + tuning is not supported.** Pick one. Tune on single-GPU; run DDP only for the final official training.
7. **Two target modes.** `tail` (forecast after input window) and `next_step` (GPT-style per-position supervision). `target_mode` lives in `features:` and must match the model registered under `model.name`.
8. **`torch.load(..., weights_only=True)` is deliberate** — do not remove it.
9. **The `portfolio/` package is an illustrative example, not a maintained lane.** It is intentionally simple. Do not extend it in lockstep with the forecasting library; users are expected to bring their own optimizer.
10. **Demo CSVs in `data/` are committed** via `.gitignore` overrides (`!data/train_features.csv` etc.). Don't `git add` other files in `data/` — they're meant to stay local.

---

## When you're unsure

- **The task doesn't match any extension row above.** Start from the lane the user named (train / tune / ensemble / benchmark / portfolio), find the entrypoint script, follow its imports into `src/`. The flow is shallow on purpose.
- **You're tempted to introduce a new abstraction.** Don't. This repo prefers concrete edit sites over frameworks. Add the regex pattern, add the dispatch branch, add the subclass. That's the pattern.
- **You're tempted to add a silent fallback.** Re-read §23 of [CONVENTION.md](CONVENTION.md). Raise instead.
- **A test fails that looks unrelated.** Check circular-import traps first; lazy-imports inside runners exist for this reason.

---

## Where to find more

| Concern | File |
|---|---|
| Repo overview, intent, quickstart | [README.md](README.md) |
| Every command / workflow | [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md) |
| Stack, naming, commands | [CLAUDE.md](CLAUDE.md) |
| Hard rules, antipatterns | [CONVENTION.md](CONVENTION.md) |
| Deferred cleanups, known gaps | [TODO.md](TODO.md) |
| Architectural ideas, alternatives | [IDEA.md](IDEA.md) |
