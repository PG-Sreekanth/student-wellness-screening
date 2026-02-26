"""
logger.py - Structured logging setup for the ML pipeline.

Provides a configured logger that writes to both console and a log file
in the artifacts/logs directory.
"""
import logging
import sys
from pathlib import Path

from config.settings import LOGS_DIR


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and return a named logger with console + file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = LOGS_DIR / "pipeline.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
