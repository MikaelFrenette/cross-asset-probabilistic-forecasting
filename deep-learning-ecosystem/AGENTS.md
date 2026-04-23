# AGENTS.md

This file is the entry point for a coding agent dropped into this repository.
Read it first — it maps every extension point a user will want to reach.

Companion docs:
- [README.md](README.md) — what the repo is and why it exists
- [CONVENTION.md](CONVENTION.md) — hard coding rules (non-negotiable)
- [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md) — every command at a glance
- [CLAUDE.md](CLAUDE.md) — stack, commands, naming

---

## What this repo does

`dlecosys` is a tabular deep-learning pipeline. Five lanes, one config object, YAML everywhere.

- **Preprocess:** a user-owned CSV / Parquet → validated, scaled, split tensors on disk
- **Train:** any `nn.Module` the user registers, single-GPU or DDP (via `torchrun`)
- **Tune:** Optuna-driven grid / random search with a pluggable validation splitter; emits a ready-to-run `best_config.yaml`
- **Ensemble:** bagging with out-of-bag validation (OOB-as-val), sample + feature bootstrapping, top-N pruning
- **Predict / Evaluate:** aggregated test predictions + metrics

Configuration is YAML with 1-level inheritance; validated by Pydantic. The same `cfg` object flows through every script.

---

## Decision tree — which lane to pick

```
User has labelled tabular data → preprocess
    │
    ├── wants one model                     → train → predict → evaluate
    ├── wants to search hyperparameters     → tune  → train (on best_config) → predict → evaluate
    └── wants a bagging ensemble            → ensemble → predict → evaluate
                                              (typically on a tuned best_config)
```

See [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md) for the exact commands.

---

## Repo map (scan once, keep in mind)

```
configs/
  template.yaml              # every knob the pipeline understands, with defaults
  demo/                      # demo configs (regression, classification, DDP, tuning, ensemble)
src/dlecosys/
  models/                    # <-- USER ADDS THEIR MODELS HERE
    mlp.py                   #     canonical "copy this pattern" demo model
    __init__.py              #     imports every user model so @register fires
  shared/                    # library code — edit to plug in new building blocks
    config/                  # PipelineConfig + section schemas + YAML loader + logging setup + builders
    data/                    # splitters, validation, synthetic, fold preprocessing, tabular transforms, sources
    models/                  # ModelFactory + @register decorator + base ModelConfig
    preprocessing/           # StandardScaler, MinMaxScaler, IntegerTokenizer artifacts
    run/                     # RunLayout, StudyLayout, EnsembleLayout, DataPaths protocol
    training/                # Trainer, DistributedTrainer, callbacks, run_training()
    tuning/                  # Optuna objective, pruning callback, study builders
    ensembling/              # BaseEnsemble, BaggingEnsemble, bootstrappers, aggregation, pruning, runner
    inference/               # Predictor (used by predict.py)
    metrics/                 # mae, mse, rmse, accuracy, binary_accuracy
scripts/
  preprocess.py              # tabular file -> validated tensor splits
  train.py                   # trains from preprocessed splits (single-GPU or DDP)
  tune.py                    # runs a hyperparameter study; emits best_config.yaml
  ensemble.py                # trains N bootstrap estimators with OOB-as-val
  predict.py                 # writes predictions.csv (single-model or ensemble-aggregated)
  evaluate.py                # computes metrics from predictions.csv
  generate_synthetic.py      # one-off demo CSV generator
tests/                       # mirrors src/ layout
outputs/                     # every run's artifacts, under <run_name>/ or <study_name>/
docs/                        # reference docs (cli_cheat_sheet.md, ...)
```

---

## "I want to..." task → edit sites

| Task | Edit | Pattern to copy |
|---|---|---|
| Add a model | `src/dlecosys/models/<name>.py` + import in `src/dlecosys/models/__init__.py` | [mlp.py](src/dlecosys/models/mlp.py) — `@register("name")` + `ModelConfig` subclass + `from_config` |
| Add a splitter | `src/dlecosys/shared/data/splitters.py` (subclass `BaseSplitter`, add to `build_splitter` dispatch) + `SplitterSection.type` Literal in `schema.py` | `KFoldSplitter` / `StratifiedKFoldSplitter` |
| Add a sample bootstrapper | `src/dlecosys/shared/ensembling/bootstrappers.py` (subclass `BaseSampleBootstrapper`, add to `build_sample_bootstrapper` dispatch) + `SampleBootstrapperSection.type` Literal in `schema.py` | `WithReplacementBootstrapper` / `NoBootstrapBootstrapper` |
| Add a feature bootstrapper | same file, subclass `BaseFeatureBootstrapper` + `build_feature_bootstrapper` dispatch + `FeatureBootstrapperSection.type` Literal | `AllFeaturesBootstrapper` / `RandomSubspaceBootstrapper` |
| Add an ensemble pruning strategy | `src/dlecosys/shared/ensembling/pruning.py::select_estimators` + `EnsemblePruningSection.strategy` Literal | `top_n` |
| Add an ensemble scheme | `src/dlecosys/shared/ensembling/<name>.py` (subclass `BaseEnsemble`) + add to `build_ensemble` dispatch + `EnsembleSection.type` Literal | `BaggingEnsemble` |
| Add a loss | `src/dlecosys/shared/config/builders.py::build_loss` + `TrainingSection.loss` Literal | existing `mse` / `cross_entropy` / `bce` branches |
| Add an optimizer | `src/dlecosys/shared/config/builders.py::_OPTIMIZERS` + `OptimizerSection.name` Literal | existing adam / sgd / adamw entries |
| Add an LR scheduler | `src/dlecosys/shared/config/builders.py::_SCHEDULERS` + `SchedulerSection.type` Literal | `cosine_annealing` / `step` / `exponential` / `reduce_on_plateau` |
| Add a metric | `src/dlecosys/shared/metrics/` + export in `__init__.py` + add to `build_metrics` in `builders.py` | `mae`, `rmse`, `accuracy` |
| Add a callback | `src/dlecosys/shared/training/callbacks.py` (subclass `Callback`) + wire into `build_callbacks` if config-driven | `EarlyStopping`, `ModelCheckpoint` |
| Add a scaler | `src/dlecosys/shared/preprocessing/scalers.py` (subclass `BaseScaler`) + add to `_SCALERS` in `tabular_transforms.py` + `DataSection.scaling` Literal | `StandardScaler`, `MinMaxScaler` |
| Add a data source format (.json, .hdf5, etc.) | `src/dlecosys/shared/data/sources.py::load_tabular` — add an extension-matching branch | existing `.csv` / `.tsv` / `.parquet` branches |
| Add an aggregation mode | `src/dlecosys/shared/ensembling/aggregation.py::aggregate` + `EnsembleSection.aggregation` Literal | `mean`, `median`, `soft_vote`, `majority` |
| Change config knobs | `src/dlecosys/shared/config/schema.py` — add a field to the appropriate section, update `template.yaml` | every existing section |

---

## Commands

All commands live in [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md).
The golden path per lane:

```bash
# Single model:  preprocess → train → predict → evaluate
# Tuning:        preprocess → tune → (winner: preprocess → train → predict → evaluate)
# Ensemble:      preprocess → ensemble → predict → evaluate
```

**Recommended full workflow:** tune the architecture first (`tuning:` config → `best_config.yaml`),
then bolt on an `ensemble:` section to the winning config and run the ensemble flow.
Do not set both `tuning:` and `ensemble:` in the same config — `preprocess.py` will reject it.

---

## Extension patterns — copy-paste templates

### Your own model

```python
# src/dlecosys/models/my_transformer.py
from pydantic import Field
from torch import nn

from dlecosys.shared.models import ModelConfig, register


class MyTransformerConfig(ModelConfig):
    input_dim: int
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    dropout: float = 0.1


@register("my_transformer")
class MyTransformer(nn.Module):
    config_class = MyTransformerConfig

    def __init__(self, input_dim, n_layers, n_heads, d_model, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)

    @classmethod
    def from_config(cls, config: MyTransformerConfig) -> "MyTransformer":
        return cls(
            input_dim=config.input_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_model=config.d_model,
            dropout=config.dropout,
        )

    def forward(self, x):
        z = self.proj(x).unsqueeze(1)
        z = self.encoder(z).squeeze(1)
        return self.head(z)
```

Then in `src/dlecosys/models/__init__.py`:

```python
from dlecosys.models import mlp  # noqa: F401
from dlecosys.models import my_transformer  # noqa: F401  ← add your import
```

Reference in config:

```yaml
model:
  name: my_transformer
  params:
    input_dim: 10
    n_layers: 3
    d_model: 128
```

### Your own splitter

```python
# In src/dlecosys/shared/data/splitters.py, alongside KFoldSplitter:

class TimeSeriesSplitter(BaseSplitter):
    """Expanding-window CV. Assumes rows are in temporal order."""

    def __init__(self, train_path, val_path=None, *, n_splits=5, seed=42):
        self.n_splits = n_splits
        super().__init__(train_path, val_path, seed=seed)

    def generate_indices(self, X, y):
        n = len(X)
        val_size = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            train_end = val_size * (k + 1)
            val_end = train_end + val_size
            yield torch.arange(train_end), torch.arange(train_end, val_end)
```

Then in the same file:

```python
# SplitterSection.type Literal in schema.py gets "time_series" added
# build_splitter gets:
if cfg.type == "time_series":
    return TimeSeriesSplitter(train_path, val_path, n_splits=cfg.n_splits, seed=cfg.seed)
```

Reference in config:

```yaml
tuning:
  splitter:
    type: time_series
    n_splits: 5
```

### Your own callback

```python
# src/dlecosys/shared/training/callbacks.py
class GradientSpikeLogger(Callback):
    """Example: warn when gradient norm jumps 10× epoch-over-epoch."""

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold
        self._prev = None

    def on_epoch_end(self, epoch, logs):
        gn = logs.get("grad_norm")
        if gn is None:
            return
        if self._prev is not None and gn > self.threshold * self._prev:
            print(f"[GradientSpikeLogger] epoch {epoch}: {gn:.2e} vs prev {self._prev:.2e}")
        self._prev = gn
```

Append to trainer callbacks via `extra_callbacks=[GradientSpikeLogger()]` in
`run_training(...)`, or wire into `build_callbacks` if config-driven.

---

## Hard rules (do not violate)

From [CONVENTION.md](CONVENTION.md):

- **§23 No silent fallbacks.** If input is invalid, raise — never impute, clip, or coerce. Users decide how to handle failures.
- **Every module has a top docstring with a `__all__`.**
- **Config is always Pydantic-validated.** No raw dicts past the YAML boundary.
- **Scripts are thin.** Logic belongs in `src/`, scripts just wire config → library → exit.
- **Logging configured once at entrypoint** via `configure_logging(cfg.logging)`.
- **Outputs directory layout is non-negotiable** — use `RunLayout` / `StudyLayout`, never construct paths manually.

---

## Gotchas that trip agents

1. **Models must be imported after definition** for `@register` to fire. Add the import to `src/dlecosys/models/__init__.py`.
2. **Config inheritance is 1 level deep.** A child config's `base:` must point to a config with `base: null`. Enforced at load time.
3. **Tuning mode writes raw splits.** When `cfg.tuning and cfg.tuning.scale_splits: true`, `preprocess.py` skips scaling and the fold loop handles it per-fold. Don't expect `scaler.pt` on disk in that case.
4. **DDP + tuning is not supported.** Pick one. Tune on single-GPU; run DDP only for the final official training.
5. **`torch.load(..., weights_only=True)` is deliberate** — do not remove it.
6. **Don't mock the database-equivalent in tests.** Load real `.pt` artifacts; the repo's tests do this via `tmp_path`.

---

## When you're unsure

- **The task doesn't match any extension row above.** Start from the lane the user named (train / tune / ensemble), find the entrypoint script, follow its imports into `src/`. The flow is shallow on purpose.
- **You're tempted to introduce a new abstraction.** Don't. This repo prefers concrete edit sites over frameworks. Add the Literal, add the dispatch branch, add the subclass. That's the pattern.
- **You're tempted to add a silent fallback.** Re-read §23 of [CONVENTION.md](CONVENTION.md). Raise instead.
- **A test fails that looks unrelated.** Check circular-import traps first (e.g. `config/__init__.py` ↔ `training/__init__.py`); we have a recurring pattern of lazy-imports in `run.py` to avoid them.

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
