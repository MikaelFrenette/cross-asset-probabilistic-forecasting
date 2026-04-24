# dlecosys

A tabular deep-learning pipeline built to be **cloned, forked, and modified directly**.
Brings together preprocessing, training (single-GPU + DDP), Optuna tuning, and
bagging ensembles behind a single YAML-driven interface.

The repo is written so a coding agent can walk in, read [AGENTS.md](AGENTS.md),
and start extending it — without inventing frameworks of its own.

---

## What you can do with it

- **Train a user-supplied `nn.Module`** on a CSV or Parquet file with one config and four commands.
- **Scale to multiple GPUs** by flipping a config flag and launching with `torchrun`.
- **Search hyperparameters** with Optuna — grid or random search, k-fold cross-validation, median pruning, automatic `best_config.yaml` emitted for the winner.
- **Build a bagging ensemble** with out-of-bag validation (each estimator's OOB set becomes its val set, so `EarlyStopping` just works), optional feature sub-sampling, and top-N pruning.
- **Extend every component** by subclassing a parent and adding one dispatch branch. No plugin discovery, no entry points, no magic.

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -e .
pip install optuna tensorboard pyarrow   # optional: tuning / logs viewer / parquet

# 2. Run the demo regression pipeline end to end
python scripts/generate_synthetic.py --task regression
python scripts/preprocess.py --config configs/demo/regression.yaml
python scripts/train.py      --config configs/demo/regression.yaml
python scripts/predict.py    --config configs/demo/regression.yaml
python scripts/evaluate.py   --config configs/demo/regression.yaml

# 3. View the training curve
tensorboard --logdir outputs/demo_regression_v1/logs/tensorboard
```

Tuning and ensemble variants are in [configs/demo/](configs/demo/); commands for
every lane are in [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md).

---

## Repo layout (bird's eye)

```
configs/     YAML configs — one source of truth, 1-level inheritance, Pydantic-validated
src/         Library code (models, data, training, tuning, ensembling, run layouts)
scripts/     Thin CLI entry points — parse args, load config, call library, exit
tests/       pytest mirror of src/ (~340 tests)
docs/        Reference docs (commands, how-tos)
outputs/     All per-run artifacts land here
```

Full map: [AGENTS.md](AGENTS.md).

---

## Design choices (the short version)

- **Clone, don't framework.** Every extension point is a dispatch branch + a Literal in the config schema. Three lines to add a new splitter / optimizer / scheduler / ensemble scheme. No registries for infrastructure.
- **Config is the contract.** One `PipelineConfig` Pydantic model. YAML at the boundary, typed objects everywhere else. Base configs inherit one level deep, never more.
- **No silent fallbacks.** If the input is wrong, raise. Don't impute, don't clip, don't coerce. The user decides how to handle failure.
- **Bring your own model.** The bundled `MLP` is a demo, not a template. Users write their own `nn.Module`, tag it `@register("name")`, and it appears in the factory.
- **Leak-free transforms.** When tuning k-fold or ensemble bagging, scaler / tokenizer fits happen per fold / per estimator — never once globally.

---

## What's in the box

| Lane | Key pieces |
|---|---|
| Preprocess | CSV / TSV / Parquet loader, train/val/test split, validation (NaN / Inf / dtype), `StandardScaler` / `MinMaxScaler`, `IntegerTokenizer`, ephemeral-vs-persisted transforms |
| Train | `Trainer` with AMP, gradient accumulation, gradient clipping; `EarlyStopping`, `ModelCheckpoint`, LR schedulers (cosine / step / exponential / plateau), optional TensorBoard |
| Distributed | `DistributedTrainer` + process-group setup, `DistributedSampler` wiring, rank-0 artifact gating |
| Tuning | Optuna `GridSampler` / `RandomSampler`, median pruner, `BaseSplitter` with `HoldoutSplitter` / `KFoldSplitter` / `StratifiedKFoldSplitter`, per-fold scaling, auto-emitted `best_config.yaml` |
| Ensembling | `BaggingEnsemble` composing a sample and feature bootstrapper, OOB-as-val training, post-training top-N pruning, aggregate-at-inference (mean / median / soft / majority) |
| Inference | `Predictor`, feature-aware ensemble aggregation, metric computation (`evaluate.py`) |

---

## Stack

- **Python 3.10+**, **PyTorch**
- **Pydantic v2** for config validation
- **Optuna** (optional, required for `tune.py`)
- **TensorBoard** (optional, required to view `tensorboard` callback output)
- **pyarrow / fastparquet** (optional, required to read `.parquet` inputs)
- **pytest** for tests

---

## For coding agents

If you're an AI assistant handed this repo, read [AGENTS.md](AGENTS.md) first.
It has the repo map, the "I want to … → edit this file" table, and the
rules that aren't obvious from the source alone.

## For everyone else

- Commands: [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md)
- Extension points + editing rules: [AGENTS.md](AGENTS.md)
- Hard rules (no silent fallbacks, etc.): [CONVENTION.md](CONVENTION.md)
- Deferred work: [TODO.md](TODO.md)
