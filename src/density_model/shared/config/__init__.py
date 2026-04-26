"""
Shared Configuration
--------------------
Logging setup + panel-pipeline config schema. Panel-specific configs
are exposed through their own module
(:mod:`density_model.shared.config.panel_schema`); only the small
``configure_logging`` helper is re-exported at this level.

Functions
---------
configure_logging
    Apply global logging configuration from a ``LoggingSection`` /
    ``PanelLoggingConfig`` at startup.
"""

from density_model.shared.config.logging import configure_logging

__all__ = ["configure_logging"]
