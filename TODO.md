# TODO

Track unfinished work, shortcuts, technical debt, and deferred cleanup here.

## Tests

- [ ] **Re-add a test suite.** The legacy tabular tests under `tests/` were
  swept in the backend rewrite. `tests/` is currently empty (just a
  `.gitkeep`). Re-add panel-pipeline coverage starting with:
  preprocessing bundle round-trip, vectorizer shapes, masked Gaussian
  NLL, model-factory registration, and predictor single-vs-ensemble
  dispatch.

## Ensembling

- [ ] **Per-estimator DDP.** `src/density_model/shared/ensembling/panel/run.py`
  trains estimators serially on a single device. The trainer
  abstractions support DDP (see `panel_distributed.py`), but the
  ensemble runner does not currently wrap each estimator in a DDP
  process group. Either start/teardown a process group per estimator,
  or split estimators across GPUs in a single launch.
- [ ] **Parallel estimator training.** Even without DDP, a process pool
  or per-GPU assignment would speed up the ensemble lane. Serial is
  the v1 default — keep it until profiled.
- [ ] **Greedy forward selection for ensemble pruning.** Today
  `pruning.py::prune_top_n` is the only strategy; `PanelEnsemblePruningConfig.strategy`
  hardcodes `"top_n"`. Greedy forward selection (Caruana 2004) often
  picks a smaller, better-performing subset; add as a new branch and
  extend the regex.

## Inputs

- [ ] **Non-CSV data sources.** `scripts/preprocess.py` calls
  `pd.read_csv` directly. Add a small dispatch on file extension
  (`.csv` / `.parquet` / `.feather`) inside the preprocess runner so
  Parquet inputs work without conversion.

## Tooling

- [ ] **Config pre-flight validation CLI.** A
  `python -m density_model.validate --config X.yaml` entrypoint that
  loads every config in this tree, resolves every registry reference
  (model name, splitter type, benchmark type, etc.), and prints a
  green/red status — caught before any GPU-time work begins.

## Template

- [ ] Describe the unfinished work or shortcut
- [ ] Describe the impact or risk
- [ ] Describe the intended follow-up
