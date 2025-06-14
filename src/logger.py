"""
Simple logging configuration module with file and console output support.

Provides a single function 'set_up_logger' that configures logging with multiple
handlers (file, console, and in-memory) using a standard Python logging library.
"""

import logging
import sys
from io import StringIO
from typing import Literal

LOG_LEVELS = {
    "DEBUG": logging.DEBUG,  # 10
    "INFO": logging.INFO,  # 20
    "WARNING": logging.WARNING,  # 30
    "ERROR": logging.ERROR,  # 40
    "CRITICAL": logging.CRITICAL,  # 50
}

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _create_handler(
    handler_class, formatter: logging.Formatter, *args, **kwargs
) -> logging.Handler:
    handler = handler_class(*args, **kwargs)
    handler.setFormatter(formatter)
    return handler


def set_up_logger(
    log_file_name: str,
    logs_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_format=DEFAULT_LOG_FORMAT,
    date_format=DEFAULT_DATE_FORMAT,
) -> StringIO:
    """
        Sets up the logger and returns log_stream for convenient access to the logs. At the same time, the logs are saved to a specified file.

        Args:
            log_file_name (str): Path to the log file
            logs_level (Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']): Level of logs to be saved to file and to be displayed in the console.
            log_format (str): Format of logs to be saved to file.
            date_format (str): Format of date in logs to be saved to file.

    Returns:
        StringIO object containing in-memory logs

    Raises:
        OSError: If a log file cannot be created or written to
    """

    try:
        # Clean existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Set up logging manually
        logger = logging.getLogger()
        logger.setLevel(LOG_LEVELS.get(logs_level, logging.INFO))

        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)

        handlers = [
            _create_handler(logging.FileHandler, formatter, log_file_name),
            _create_handler(logging.StreamHandler, formatter, sys.stdout),
        ]

        # in memory logs to be saved as output
        log_stream = StringIO()
        handlers.append(_create_handler(logging.StreamHandler, formatter, log_stream))

        return log_stream

    except OSError as e:
        raise OSError(f"Failed to set up logging: {e}")
