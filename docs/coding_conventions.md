# Coding Conventions

This repository is intended to be public-facing. Python modules should be written to be readable, explicit, and easy to document.

## Module Requirements

Every Python module must start with a top-of-file documentation block describing what the file contains and what it does.

Example:

```python
"""
FRED Client
-----------
HTTP client and typed payload helpers for FRED category access.
"""
```

## Public API Requirements

Every publicly exposed class must define:

- A scikit-learn style docstring using `Parameters`, `Attributes`, `Returns`, `Raises`, and `Examples` sections where relevant.
- Explicit exports via `__all__`.

Every publicly exposed function or method must define:

- A scikit-learn style docstring.
- Parameter and return value documentation when applicable.

Private helpers do not need full public-facing documentation, but they should still be clear and concise.

## Input Validation

Use `pydantic` for user-facing input validation, including repository configuration loaded from YAML and any public request objects that accept external input.

Validation must fail fast:

- Do not add fallback behavior for invalid values.
- Do not silently replace invalid user input with defaults outside the declared schema.
- Do not auto-correct semantically invalid inputs unless that transformation is explicitly part of the documented schema.

If an input does not satisfy the `pydantic` model, raise the validation error and stop.

## Export Rules

Any Python module containing classes must define `__all__` near the top of the file after imports.

Example:

```python
__all__ = ["FredCategory", "FredRelease", "FredSeries", "FredSource", "FredClient"]
```

Only include names that are intentionally part of the public API.

## Docstring Style

Use the scikit-learn / numpydoc structure.

Example:

```python
"""
Volatility Window
-----------------
Container definitions and validation helpers for rolling volatility windows.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["VolatilityWindow"]


@dataclass(slots=True)
class VolatilityWindow:
    """
    Metadata for a rolling volatility estimation window.

    Parameters
    ----------
    start_index : int
        Inclusive starting index of the window.
    end_index : int
        Exclusive ending index of the window.
    annualization_factor : float, default=252.0
        Scaling factor used to annualize realized volatility.
    """

    start_index: int
    end_index: int
    annualization_factor: float = 252.0
```

## Repository-Level Guidance

- Prefer explicit imports over wildcard imports.
- Keep modules focused on one responsibility.
- Use type hints for public interfaces.
- Use `pydantic` models for externally supplied configuration and request payloads.
- Treat documentation as part of the implementation, not as optional cleanup.
- For backbone infrastructure such as calendar handling, prefer explicit backend
  abstractions over hard-coding one third-party package throughout the repo.

## YAML Configuration Files

Public YAML configuration files under `configs/` must be documented inline.

- Add concise comments explaining what each meaningful knob does.
- Prefer comments directly above the field they describe.
- Treat example YAML files as public-facing documentation, not just machine input.
- When adding a new public config field, update the corresponding YAML example immediately so users can understand it without opening Python code.
