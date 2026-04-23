import logging

import pytest
from pydantic import ValidationError

from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.schema import LoggingSection


class TestLoggingSection:
    def test_defaults(self):
        cfg = LoggingSection()
        assert cfg.level == "INFO"
        assert cfg.format == "text"
        assert cfg.include_timestamps is True

    def test_valid_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            cfg = LoggingSection(level=level)
            assert cfg.level == level

    def test_invalid_level_raises(self):
        with pytest.raises(ValidationError):
            LoggingSection(level="VERBOSE")

    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError):
            LoggingSection(format="json")

    def test_timestamps_optional(self):
        cfg = LoggingSection(include_timestamps=False)
        assert cfg.include_timestamps is False


class TestConfigureLogging:
    def test_sets_root_level_info(self):
        configure_logging(LoggingSection(level="INFO"))
        assert logging.getLogger().level == logging.INFO

    def test_sets_root_level_debug(self):
        configure_logging(LoggingSection(level="DEBUG"))
        assert logging.getLogger().level == logging.DEBUG

    def test_sets_root_level_warning(self):
        configure_logging(LoggingSection(level="WARNING"))
        assert logging.getLogger().level == logging.WARNING

    def test_idempotent(self):
        configure_logging(LoggingSection(level="ERROR"))
        configure_logging(LoggingSection(level="INFO"))
        assert logging.getLogger().level == logging.INFO
