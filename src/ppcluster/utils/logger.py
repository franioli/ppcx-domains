# -----------------------------------------------------------------------------
#
#  Python dual-logging setup (console and log file),supporting different log
#  levels and colorized output
#
#
#  Based on:
#  https://stackoverflow.com/a/13733863/1976617
#  https://uran198.github.io/en/python/2016/07/12/colorful-python-logging.html
#  https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
#
#  Francesco Ioli
#  2026/01/19
#
# -----------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import logging
import sys
from datetime import datetime
from pathlib import Path

# Constants
DEFAULT_FORMAT = "%(asctime)s | [%(levelname)-8s] %(message)s"
DEBUG_FORMAT = "%(asctime)s | [%(filename)s -> %(funcName)s], line %(lineno)d - [%(levelname)-8s] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes
COLORS = {
    "DEBUG": "\033[0;36m",  # Cyan
    "INFO": "\033[0;32m",  # Green
    "WARNING": "\033[0;33m",  # Yellow
    "ERROR": "\033[0;31m",  # Red
    "CRITICAL": "\033[0;37m\033[41m",  # White on Red
    "RESET": "\033[0m",  # Reset
}


def get_logger(name: str | None = None) -> logging.Logger:
    """Return an existing logger (do not create handlers)."""
    return logging.getLogger(name)


def setup_logger(
    level: str | int | None = logging.INFO,
    name: str | None = None,
    log_to_file: bool = False,
    log_folder: Path | str | None = None,
    redirect_to_stdout: bool = False,
    force: bool = True,
) -> logging.Logger:
    """Setup and configure logger with color support.

    Args:
        name: Logger name (root logger if None)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file: Whether to log to file
        log_folder: Path to logs directory (default: ./logs)
        redirect_to_stdout: Redirect console output to stdout instead of stderr
        force: If true, any existing handlers attached to the root logger are removed and closed, before carrying out the configuration as specified by the other arguments.

    Returns:
        Configured logger instance
    """

    # Normalize level (avoid using private _nameToLevel if possible)
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), None)
        if isinstance(lvl, int):
            level = lvl
        else:
            try:
                level = int(level)
            except Exception:
                level = logging.INFO

    # Check if debug level is set
    debug = level == logging.DEBUG

    # If force is True, remove existing handlers from the root logger
    if force:
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

    # target logger (root if name is None)
    logger = logging.getLogger(name)

    # If logger is already configured and we are not forcing, return it as is.
    # If the logger has any non-NullHandler, treat it as already configured and return.
    if (
        not force
        and logger.handlers
        and any(not isinstance(h, logging.NullHandler) for h in logger.handlers)
    ):
        return logger

    # If forcing, remove/close handlers from target logger only
    if force and logger.handlers:
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()

    # Configure level and propagation behavior
    logger.setLevel(level=level)  # type: ignore
    if name is not None:
        # When we add handlers to a named logger, avoid bubbling to parent,
        # which prevents duplicate logs when both root and child have handlers.
        logger.propagate = False

    # Create console handler with color support
    stream = sys.stdout if redirect_to_stdout else sys.stderr
    console = logging.StreamHandler(stream)
    console.setLevel(level)  # type: ignore
    console.setFormatter(
        ColoredFormatter(
            DEBUG_FORMAT if level == logging.DEBUG else DEFAULT_FORMAT, datefmt=DATE_FMT
        )
    )
    logger.addHandler(console)

    # Optional file logging
    if log_to_file:
        log_folder = Path(log_folder or "./logs")
        log_folder.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"log_{timestamp}.log"
        log_file = log_folder / log_name

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                DEBUG_FORMAT if debug else DEFAULT_FORMAT, datefmt=DATE_FMT
            )
        )
        logger.addHandler(file_handler)

    return logger


def set_log_level(logger_name: str, level: str | int) -> logging.Logger:
    """Change logger level."""
    logger = logging.getLogger(logger_name)

    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)  # type: ignore

    logger.setLevel(level)

    return logger


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to levelname field."""

    def format(self, record) -> str:
        # Add color to levelname if running in terminal
        if sys.stderr.isatty():
            color = COLORS.get(record.levelname, COLORS["RESET"])
            record.levelname = f"{color}{record.levelname}{COLORS['RESET']}"
        return super().format(record)
