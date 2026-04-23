# TODO

Track unfinished work, shortcuts, technical debt, and deferred cleanup here.

## Ensembling

- [ ] **Per-estimator DDP** — `ensemble.distributed.enabled: true` currently raises `NotImplementedError` in `run_ensemble`. The existing `setup/teardown` model assumes one process-group lifecycle per invocation, not one per estimator. Needs estimator-scoped process group setup/teardown inside the loop, or a wrapper that re-inits cleanly.
- [ ] **Parallel estimator training** — serial is the v1 default. Process pool or multi-GPU sharding (assign estimators to devices) would speed up large ensembles. NN parallelization is tricky; defer until profiled.
- [ ] **Greedy forward selection** for ensemble pruning (Caruana 2004) — currently only `strategy: top_n` is implemented. Greedy often selects a smaller, better-performing subset than top-N. Add as a new strategy branch in `pruning.py::select_estimators` and a Literal entry in `EnsemblePruningSection.strategy`.

## Other

- [ ] **Non-CSV data sources** — `preprocess.py` hardcodes `pd.read_csv`. Wire `BaseDataModule` in for Parquet / DB / streaming inputs.
- [ ] **Config pre-flight validation CLI** — `dlecosys validate --config X.yaml` that resolves every registry reference before any expensive work.

## Template

- [ ] Describe the unfinished work or shortcut
- [ ] Describe the impact or risk
- [ ] Describe the intended follow-up
