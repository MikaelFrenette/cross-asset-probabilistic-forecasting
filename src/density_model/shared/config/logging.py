"""
Logging Setup
-------------
Configures the root logger once from a validated logging-section config
(``PanelLoggingConfig`` from
:mod:`density_model.shared.config.panel_schema`, or any object with
``level``, ``include_timestamps`` attributes).

Functions
---------
configure_logging
    Apply global logging configuration at application startup.
"""

from __future__ import annotations

import logging as _logging
from typing import Any

__all__ = ["configure_logging"]


def configure_logging(cfg: Any) -> None:
    """Configure the root logger from a validated panel logging config.

    Called once at the application entrypoint, before any library code
    emits log records. ``cfg`` must expose ``level`` (str) and
    ``include_timestamps`` (bool); the panel-pipeline configs all do.
    """
    level = getattr(_logging, cfg.level)
    if cfg.include_timestamps:
        fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%(levelname)-8s %(name)s — %(message)s"
        datefmt = None
    _logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)
