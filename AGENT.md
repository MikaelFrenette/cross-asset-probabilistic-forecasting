# AGENT.md

This file defines repository-specific instructions for coding agents working in this project.

## Purpose

The goal of this repository is to build a public-facing Python codebase for volatility modeling, including cross-attention models and supporting data, training, and evaluation utilities.

Agents should optimize for:

- clear public APIs,
- reproducible research workflows,
- readable implementation,
- minimal hidden behavior,
- documentation that is useful to human contributors.

## Source Of Truth

For Python style and documentation requirements, follow [docs/coding_conventions.md](docs/coding_conventions.md).

If there is a conflict:

1. explicit user instructions override this file,
2. this file overrides agent defaults for this repository,
3. `docs/coding_conventions.md` governs Python coding style and public documentation.

## Repository Expectations

- Treat this as a public repository, not a scratch workspace.
- Prefer small, explicit modules over notebook-style scripts.
- Keep modeling code, data utilities, training logic, and evaluation logic separated.
- Treat the public workflow as three separate stages:
  - Yahoo extraction to an inspectable CSV,
  - CSV-based training,
  - manifest-based prediction.
- For model development, keep train, validation, and prediction/backtest datasets as separate extracted artifacts rather than embedding date-split logic deep inside `scripts/train.py`.
- Keep the calendar layer flexible and backend-agnostic where possible. Avoid
  binding historical support to one restrictive third-party package when a
  broader or approximate fallback is available and clearly signaled.
- Avoid introducing hidden side effects at import time.
- Use typed public interfaces.
- Keep filenames descriptive and stable.
- Fail fast on invalid user input or configuration. Do not add fallback behavior that silently changes requested behavior.

## Python Implementation Rules

- Every Python file must begin with a top-of-file documentation block.
- Every module defining public classes must declare `__all__`.
- Every public class, function, and method must use a scikit-learn / numpydoc-style docstring.
- Use `pydantic` models for user-provided inputs and repository configuration objects.
- Do not coerce invalid user input through permissive fallback logic when `pydantic` validation should reject it.
- Prefer dataclasses for lightweight typed containers.
- Prefer explicit imports over wildcard imports.
- Avoid overly clever abstractions in the initial public layout.

## Package Design Guidance

When adding new code, prefer a structure like:

- `src/.../data/` for dataset loading, transforms, and batching,
- `src/.../models/` for neural architectures,
- `src/.../training/` for optimization and training loops,
- `src/.../evaluation/` for metrics and backtests,
- `src/.../config/` for typed configuration objects,
- `tests/` for unit and integration coverage.

Do not collapse all responsibilities into a single module.

For this repository, use `src/density_model/` as the main package root unless the user requests a different public package name.

## Public Workflow Contracts

- `scripts/extract_features.py` should be responsible for Yahoo download, configured market-calendar normalization, return construction, and saving an inspectable raw feature CSV.
- `scripts/train.py` should consume explicit train and validation feature CSVs, fit preprocessing on the training CSV only, train the model, and save artifacts plus a frozen training manifest.
- `scripts/predict.py` should consume a feature CSV plus the frozen training manifest. It must not depend on a mutable training YAML file.
- Public runnable entrypoints should live under `scripts/`, with any root-level wrappers kept thin and backward-compatible.
- Prediction inputs should match the public feature schema, not pre-tokenized or pre-scaled internal tensors.
- If new workflow config files are added, keep them separated by responsibility rather than folding them back into one monolithic config.

## Documentation Expectations

- Public modules should be understandable from the module docstring and exported names.
- Add concise comments only where the code is not self-explanatory.
- Prefer documentation that explains intent and interface, not obvious mechanics.
- Public YAML files under `configs/` should include inline comments describing what each important knob does.
- Treat example config files as user-facing documentation and keep them readable without requiring a code dive.

## Editing Behavior

- Do not remove user-authored changes unless explicitly requested.
- Preserve existing conventions once established.
- When creating new public modules, include the full required documentation structure immediately rather than leaving it for later cleanup.
- Prefer incremental, reviewable changes over large speculative rewrites.

## Testing Expectations

- Add or update pytest coverage for new public behavior.
- Prefer deterministic unit tests that do not depend on live network access.
- Mock external data providers such as Yahoo Finance in tests.
- Keep parsing, validation, and transformation logic testable without downloading real data.

## When Unsure

- Choose the more explicit design.
- Choose the simpler public API.
- Ask whether a module is intended to be public or internal before expanding exports broadly.
