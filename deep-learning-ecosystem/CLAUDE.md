# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

- **[AGENTS.md](AGENTS.md)** — repo map, "task → edit site" table, copy-paste templates for adding a model / splitter / callback. Start here when you are asked to extend the pipeline.
- **[CONVENTION.md](CONVENTION.md)** — hard rules. Non-negotiable.

## Conventions

All coding agent behavior is governed by [CONVENTION.md](CONVENTION.md). Read it before writing any code. The rules there are requirements, not suggestions.

## Stack

- **Python** for application and library code
- **Rust** for performance-sensitive or reliability-critical components
- **pytest** for all tests
- **Pydantic** for config objects, schemas, and validated domain payloads — not raw dicts
- **YAML** (`.yaml`, never `.yml`) for all configuration files

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/path/to/test_file.py

# Run a single test by name
pytest tests/path/to/test_file.py::test_function_name

# Run tests with stdout visible
pytest -s

# Run tests with coverage report
pytest --cov=src
```

## Repository Structure

```
data/       # datasets, raw inputs, intermediate artifacts
configs/    # YAML configuration files
outputs/    # generated outputs, reports, logs, run artifacts
src/        # library and application source code
tests/      # automated tests
scripts/    # thin operational entrypoints
```

`src/` holds one primary package organized by domain:

```
src/
    project_name/
        __init__.py
        domain_a/
        domain_b/
        shared/     # only genuinely cross-domain code
```

`tests/` mirrors the relevant source structure. Pick one layout and stay consistent:

```
tests/unit/ + tests/integration/
# or
tests/forecasting/ + tests/validation/ + ...
```

## Key Architectural Rules

**Every Python module** requires a top-of-file docstring listing its purpose and public classes, plus an explicit `__all__ = []`.

**Config is always Pydantic-validated.** Load YAML at the application boundary into a Pydantic model; pass the config object down through constructors. Never scatter hardcoded constants across modules.

**Scripts stay thin.** Business logic lives in `src/`. Scripts parse inputs, load config, call library code, and exit.

**Logging is configured once** at the entrypoint from YAML config. Pass a logger to constructors only for reusable services with meaningful operational logging; use a module-level `logging.getLogger(__name__)` elsewhere.

**Config tree shape:** each major subdirectory under `configs/` has a `base.yaml` with shared defaults; environment- or task-specific files override only what differs.

## Naming

| Artifact | Convention |
|---|---|
| Packages / modules | `lowercase_underscores` — no `utils.py`, `helpers.py`, `misc.py` |
| Classes | `PascalCase` reflecting a single responsibility |
| Functions / methods | `snake_case` describing an action or transformation |
| Constants | `UPPER_SNAKE_CASE` near their owning module |
| Test files | `test_*.py`, mirroring the module under test |
| Scripts | `lowercase_underscores`, verb-first, grouped in subdirectories |
| Config files | `lowercase_underscores`, purpose-communicating names |

## Project Tracking

- **[TODO.md](TODO.md)** — record every shortcut, deferred cleanup, and known gap here immediately when taken
- **[IDEA.md](IDEA.md)** — capture architectural ideas and alternatives before they are lost
