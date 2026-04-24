# Coding Agent Conventions

This document defines the expected coding behavior for coding agents working in this repository.

## 1. Python Module Header Documentation

Every Python script must begin with a top-of-file module docstring that explains:

- the module's purpose
- the public classes it defines
- the responsibilities of those classes

Use this structure as the default template:

```python
"""
Forecasting Dataset
-------------------
High-level dataset classes for time-series forecasting tasks.

Classes
-------
ForecastingDataset
    Combines canonicalized time-series inputs with forecasting configuration.
    Supports 2D lag-matrix generation for feature engineering.

SequentialDataset
    Builds 3D forecasting sequences (N, T, K) without ID vectorization.

VectorizedPanelDataset
    Builds 4D forecasting sequences (B, I, T, K) with per-ID alignment and padding.

These classes provide a unified preprocessing interface for forecasting models,
bridging static and dynamic features with configurable lag specifications and
window assembly strategies.
"""
```

This header is required for every Python module, even when the classes differ from the example above.

## 2. `__all__` Is Required

Every Python script must define `__all__ = []` explicitly.

Rules:

- update `__all__` to expose intended public symbols
- keep private implementation details out of `__all__`
- do not omit `__all__`, even for small modules

Base requirement:

```python
__all__ = []
```

## 3. Public API Docstrings

Every public class and every dedicated public API must have a docstring.

Requirements:

- use the scikit-learn docstring style for public classes and public functions
- describe purpose, parameters, attributes, returns, and notes where applicable
- include examples when the API is non-obvious

For private classes, use a minimal simplified docstring such as:

```python
"""Internal helper."""
```

For private methods and small internal functions, keep docstrings brief and only add them when they clarify non-obvious behavior.

## 4. Prefer Pydantic

Use Pydantic as much as possible for structured data and validated interfaces.

Expected usage:

- configuration objects
- request and response schemas
- validated domain payloads
- boundaries between modules and external systems

Avoid passing around loosely structured dictionaries when a Pydantic model would make contracts clearer.

## 5. Prefer Pytest and Rust

Testing should default to `pytest`.

Guidelines:

- write pytest tests for new behavior
- add regression tests for bug fixes
- keep tests narrow and behavior-focused

Use Rust as much as possible where it meaningfully improves:

- correctness
- performance
- safety
- maintainability of critical logic

Python remains acceptable where it is the better fit, but performance-sensitive or reliability-critical components should strongly consider Rust.

## 6. Class Design Expectations

Classes must have clear boundaries and narrow responsibilities.

Rules:

- do not create do-it-all classes
- prefer reusable classes over one-off orchestration objects
- keep data modeling, transformation, persistence, and orchestration separate
- design classes so they can be tested in isolation
- favor composition over bloated inheritance trees

When responsibilities start to mix, split the class.

## 7. Project Tracking Files

The repository should maintain the following root-level files:

- `TODO.md`: unfinished work, deferred cleanup, shortcuts, and known gaps
- `IDEA.md`: brainstorming, exploratory notes, and ideas worth preserving

### `TODO.md`

Use `TODO.md` to track:

- unfinished implementation work
- temporary shortcuts
- technical debt introduced intentionally
- follow-up validation or refactoring work

If an agent takes a shortcut, it must be recorded in `TODO.md`.

### `IDEA.md`

Use `IDEA.md` for:

- brainstorming on the fly
- architectural ideas
- possible future improvements
- alternative approaches worth revisiting

Keep ideas lightweight, but write them down before they are lost.

## 8. Default Operating Principle

When writing code in this repository, agents should optimize for:

- explicit interfaces
- validated data contracts
- reusable abstractions
- testability
- maintainable boundaries

If there is a tradeoff, prefer clarity and long-term maintainability over short-term convenience.

## 9. Standard Repository Structure

The repository must follow this standard top-level structure:

```text
data/
configs/
outputs/
src/
tests/
scripts/
```

Expectations:

- `data/` stores datasets, raw inputs, intermediate artifacts, or local data assets
- `configs/` stores configuration files
- `outputs/` stores generated outputs, reports, logs, and run artifacts
- `src/` stores application and library source code
- `tests/` stores automated tests
- `scripts/` stores operational scripts, utilities, and entrypoints

When introducing new work, place files in the appropriate directory rather than inventing ad hoc top-level folders.

## 10. Structured Python Package Layout

Python code must be organized in a standard, structured way using subdirectories with clear ownership boundaries.

Rules:

- avoid dumping many unrelated Python files into a single flat directory
- group modules by domain, feature, or responsibility
- keep `src/` organized as packages and subpackages
- mirror important source structure in `tests/` where practical
- place executable or workflow-oriented scripts in `scripts/`, not inside core library packages unless there is a clear reason

Preferred direction:

```text
src/
    project_name/
        domain_a/
        domain_b/
        shared/

tests/
    domain_a/
    domain_b/
    shared/

scripts/
    training/
    evaluation/
    maintenance/
```

The goal is explicit structure, clear module boundaries, and discoverable code organization.

## 11. Naming Conventions

Use consistent naming across source code, tests, scripts, and configuration files.

### Python Packages and Modules

Rules:

- package names must be lowercase
- module filenames must be lowercase with underscores when needed
- avoid vague names like `utils.py`, `helpers.py`, `misc.py`, or `stuff.py` unless the scope is genuinely narrow and local
- prefer domain-specific names such as `forecast_config.py`, `panel_alignment.py`, or `dataset_registry.py`

Examples:

```text
src/project_name/data_loading/
src/project_name/forecasting/panel_alignment.py
src/project_name/validation/schema_models.py
```

### Classes

Rules:

- class names must use `PascalCase`
- names should reflect a single responsibility
- avoid generic names like `Manager`, `Processor`, or `Handler` unless the role is truly precise in context

Prefer:

- `ForecastConfig`
- `PanelSequenceBuilder`
- `DatasetCanonicalizer`

Over:

- `DataManager`
- `MainProcessor`
- `UtilityHandler`

### Functions and Methods

Rules:

- function and method names must use `snake_case`
- names should describe an action or transformation
- boolean-returning functions should read like predicates where possible

Prefer:

- `build_feature_matrix`
- `validate_config`
- `is_aligned_panel`

### Constants

Rules:

- constants must use `UPPER_SNAKE_CASE`
- keep constants near the module or domain they belong to
- avoid centralizing unrelated constants into a single catch-all file

### Test Files

Test files should follow `pytest` conventions.

Rules:

- test filenames must start with `test_`
- test functions must start with `test_`
- group tests by module or behavior
- mirror the source tree where practical

Examples:

```text
tests/forecasting/test_panel_alignment.py
tests/validation/test_schema_models.py
```

### Script Files

Scripts in `scripts/` should be named for their operational purpose.

Rules:

- use lowercase filenames with underscores
- prefer verbs or explicit workflow names
- group scripts into subdirectories when there are multiple workflows

Examples:

```text
scripts/training/run_training.py
scripts/evaluation/generate_forecast_report.py
scripts/maintenance/rebuild_feature_store.py
```

### Configuration Files

Configuration files in `configs/` should be structured and discoverable.

Rules:

- use lowercase filenames with underscores or hyphens consistently within a subdomain
- group configs by workflow, environment, or subsystem
- file names should communicate purpose directly

Examples:

```text
configs/training/base.yaml
configs/training/production.yaml
configs/forecasting/panel_monthly.yaml
configs/features/static_features.yaml
```

## 12. `src/` Package Shape

The `src/` directory should contain one primary project package unless there is a clear multi-package reason not to.

Preferred pattern:

```text
src/
    project_name/
        __init__.py
        domain_a/
        domain_b/
        shared/
```

Guidelines:

- avoid placing loose `.py` files directly under `src/` unless they are package bootstrap files
- use `shared/` only for genuinely cross-domain code
- avoid turning `shared/` into a dumping ground
- keep domain packages cohesive and bounded

## 13. Test Layout

Tests should be organized to make ownership and coverage obvious.

Rules:

- mirror the relevant `src/` structure when possible
- keep unit tests close in naming to the module they validate
- separate broader integration or end-to-end tests when needed

Preferred direction:

```text
tests/
    unit/
    integration/
    project_name/
```

or:

```text
tests/
    forecasting/
    validation/
    pipelines/
```

Choose one coherent approach and stay consistent.

## 14. Script Boundaries

Scripts must stay thin.

Rules:

- business logic belongs in `src/`, not embedded inside scripts
- scripts should parse inputs, load config, call library code, and exit cleanly
- if a script grows substantial logic, move that logic into reusable modules

This keeps scripts operational and source packages reusable.

## 15. Config-Driven Repository

This repository must be config-driven.

Rules:

- runtime behavior should be controlled by configuration rather than hardcoded values
- experiments, pipelines, environments, and feature switches should be expressed through config files
- code should consume validated config objects rather than scattering inline constants across modules
- when practical, load configuration into Pydantic models at the application boundary

Hardcoded defaults are acceptable only when they are truly local implementation details and not part of the operational contract.

## 16. YAML-Only Config Files

All configuration files must use the `.yaml` extension.

Rules:

- store configs under `configs/`
- use `.yaml`, not `.yml`
- do not introduce alternative config formats unless there is a strong technical reason
- config filenames must clearly communicate scope and purpose

Examples:

```text
configs/training/base.yaml
configs/training/production.yaml
configs/forecasting/panel_monthly.yaml
```

## 17. Structured YAML Tree

The YAML configuration tree must be well structured and predictable.

Rules:

- organize config files by domain, workflow, or subsystem
- avoid a flat pile of unrelated YAML files under `configs/`
- shared configuration for a domain should live in `base.yaml`
- environment-specific or task-specific files should extend, specialize, or override the shared base configuration in a consistent way

Preferred direction:

```text
configs/
    training/
        base.yaml
        development.yaml
        production.yaml
    forecasting/
        base.yaml
        monthly.yaml
        quarterly.yaml
    features/
        base.yaml
        static.yaml
        dynamic.yaml
```

## 18. Base Config Rules

Each major config subtree should define a `base.yaml` that contains the common defaults for that area.

`base.yaml` should contain:

- shared defaults
- common paths
- stable feature toggles
- default model or pipeline settings
- values intended to be inherited by more specific configs

Specialized config files should contain only the values that differ from the base, plus any settings unique to that specific run mode or workflow.

Avoid duplicating the same values across multiple YAML files when they belong in `base.yaml`.

## 19. Logging and Controllable Verbosity

The repository should use structured, consistent logging with controllable verbosity.

Rules:

- use a logger for operational visibility instead of ad hoc `print()` statements
- logging verbosity must be configurable
- default logging behavior should be defined in YAML config
- the application entrypoint should initialize logging once and pass configured dependencies downward in a consistent way

Preferred verbosity settings include:

- `DEBUG`
- `INFO`
- `WARNING`
- `ERROR`

## 20. Logger Ownership and Injection

Logging configuration should be controlled at the application boundary, not hardcoded deep inside implementation classes.

Preferred approach:

- configure logging in the entrypoint, runner, or top-level service assembly
- pass either a logger or a validated logging/config object into constructors where a class has meaningful operational logging needs
- for simple modules, using a module-level logger is acceptable if it still respects global configured verbosity

Constructor injection is appropriate when:

- the class is a reusable service
- the class is instantiated in multiple contexts
- the class needs explicit control over logging behavior or contextual naming

Avoid:

- each class configuring logging on its own
- hidden logger setup inside lower-level modules
- hardcoded verbosity levels scattered across the codebase

## 21. Logging Config in YAML

Logging settings should live in config and be validated through a Pydantic model when practical.

Preferred direction:

```yaml
logging:
  level: INFO
  format: text
  include_timestamps: true
```

Typical responsibilities:

- `base.yaml` defines standard logging defaults
- environment- or workflow-specific YAML files override verbosity when needed
- application startup loads and validates logging config before constructing services

## 22. Practical Guidance

As a default rule:

- pass config objects to constructors
- pass a logger to constructors only when that class has non-trivial operational logging needs
- otherwise, use a module-level logger that inherits the configured global behavior

This keeps class APIs clean while still allowing explicit logging control where it matters.

### Canonical Example

`configs/base.yaml`

```yaml
logging:
  level: INFO
  format: text
  include_timestamps: true
```

`src/project_name/config/logging.py`

```python
"""
Logging Configuration
---------------------
Validated logging configuration models for application startup.

Classes
-------
LoggingConfig
    Stores logging verbosity and formatting options loaded from YAML.
"""

from pydantic import BaseModel

__all__ = ["LoggingConfig"]


class LoggingConfig(BaseModel):
    """Validated logging settings used during application bootstrap.

    Parameters
    ----------
    level : str, default="INFO"
        Logging verbosity level.
    format : str, default="text"
        Output format for emitted logs.
    include_timestamps : bool, default=True
        Whether timestamps are included in log output.
    """

    level: str = "INFO"
    format: str = "text"
    include_timestamps: bool = True
```

`scripts/run_pipeline.py`

```python
"""
Pipeline Runner
---------------
Application entrypoint for loading config, initializing logging, and running
the main pipeline.

Classes
-------
This module does not define public classes.
"""

import logging

__all__ = []


def configure_logging(logging_config) -> None:
    """Configure global logging from validated configuration.

    Parameters
    ----------
    logging_config : LoggingConfig
        Validated logging settings.
    """

    logging.basicConfig(level=getattr(logging, logging_config.level))


def main() -> None:
    """Run the configured application pipeline."""

    # Load YAML, validate with Pydantic, then initialize logging once.
    ...
```

`src/project_name/pipeline/service.py`

```python
"""
Pipeline Service
----------------
Reusable pipeline service for orchestrating application work.

Classes
-------
PipelineService
    Coordinates the main pipeline using validated dependencies.
"""

import logging

__all__ = ["PipelineService"]

LOGGER = logging.getLogger(__name__)


class PipelineService:
    """Coordinate pipeline execution.

    Parameters
    ----------
    config : PipelineConfig
        Validated pipeline configuration.
    logger : logging.Logger or None, default=None
        Optional injected logger for operational visibility.
    """

    def __init__(self, config, logger=None) -> None:
        self.config = config
        self.logger = logger or LOGGER
```

This example illustrates the preferred pattern:

- YAML defines verbosity
- Pydantic validates config
- the entrypoint configures logging once
- reusable services may accept an optional logger
- lower-level modules do not configure logging themselves

## 23. No Silent Fallbacks

Fallback behaviors must never be implemented unless explicitly requested by the user.

Rules:

- do not silently impute, fill, clip, coerce, or drop data
- do not catch exceptions and substitute defaults
- do not retry operations with degraded settings
- do not cast types without an explicit instruction to do so
- if something is wrong, raise a clear and specific error

Silent fallbacks mislead experiments: bad data passes through the pipeline
looking like good data, and failures surface far from their root cause.
When an unexpected condition is reached, the code must stop and surface it.
The user decides how to handle it — not the code.

## 24. Extension Sites Over Frameworks

This repository is designed to be cloned and edited directly. Do not invent
registries, plugin systems, or indirection layers when a concrete edit site
will do.

Rules:

- Every extension point already has a dispatch function (`build_splitter`,
  `build_sample_bootstrapper`, `build_ensemble`, `build_scheduler`,
  `build_loss`, ...) and a corresponding `Literal` in a Pydantic section.
- To add a new variant: subclass the parent (or write the function), add
  the branch to the dispatch, add the string to the Literal. Three edits,
  one lookup table.
- Do not introduce entry-point discovery, runtime plugin loading, or
  auto-wiring schemes. They add complexity without benefit in a repo
  meant to be forked and modified in place.

The single exception is models, which use an `@register("name")` decorator
+ registry. That's because user architectures are the most frequently
added component and the decorator keeps them self-contained in their own
file — everything else lives in one or two central library modules.

See [AGENTS.md](AGENTS.md) for the complete task → edit-site map.

## 25. Entry Point for Coding Agents

When a coding agent is handed this repository, it should:

1. Read [AGENTS.md](AGENTS.md) for the repo map and "task → edit site" table.
2. Read this file for hard rules (especially §23 and §24).
3. Read [docs/cli_cheat_sheet.md](docs/cli_cheat_sheet.md) for the command surface.
4. Only then start editing code.

Do not infer architecture from the source tree alone. The docs above
codify intent that cannot be recovered by reading code in isolation.
