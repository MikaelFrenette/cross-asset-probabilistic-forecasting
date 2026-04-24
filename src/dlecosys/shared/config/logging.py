"""
Logging Setup
-------------
Configures the root logger once from a validated LoggingSection.

Functions
---------
configure_logging
    Apply global logging configuration at application startup.
"""

from __future__ import annotations

import logging as _logging

from dlecosys.shared.config.schema import LoggingSection

__all__ = ["configure_logging"]


def configure_logging(cfg: LoggingSection) -> None:
    """Configure the root logger from a validated LoggingSection.

    Must be called once at the application entrypoint, before any library
    code emits log records.
    """
    level = getattr(_logging, cfg.level)
    if cfg.include_timestamps:
        fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%(levelname)-8s %(name)s — %(message)s"
        datefmt = None
    _logging.basicConfig(level=level, format=fmt, datefmt=datefmt, force=True)
