"""Structured logging configuration for the Japan Grid Pipeline.

Provides a consistent logging setup with structured output for traceability
across all pipeline stages (download, parse, standardize, convert, export).
"""

import logging
import sys
from typing import Optional


# Default log format with timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Pipeline-specific logger name prefix
LOGGER_PREFIX = "japan_grid"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = DEFAULT_FORMAT,
    date_fmt: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Configure structured logging for the pipeline.

    Sets up the root pipeline logger with console output and optional file output.
    All pipeline modules should use ``get_logger(__name__)`` to inherit this
    configuration.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to a log file. If provided, logs are written
            to both console and file.
        fmt: Log message format string.
        date_fmt: Date format string for log timestamps.

    Returns:
        The configured root pipeline logger.
    """
    logger = logging.getLogger(LOGGER_PREFIX)
    logger.setLevel(level)

    # Avoid adding duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # Console handler (stderr for visibility alongside stdout data output)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the pipeline logger hierarchy.

    Usage in modules::

        from src.utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing region: %s", region_name)

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        A logger that inherits the pipeline logging configuration.
    """
    if name.startswith(LOGGER_PREFIX):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_PREFIX}.{name}")
