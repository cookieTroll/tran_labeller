import logging
import os
from io import StringIO

import pytest

from src.logger import LOG_LEVELS, set_up_logger


@pytest.fixture
def temp_log_file(tmp_path):
    """Fixture to create a temporary log file path"""
    log_file = tmp_path / "test.log"
    yield str(log_file)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def clean_logger():
    """Fixture to ensure logger is clean before each test"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    yield
    # Clean up after test
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def test_basic_setup(temp_log_file, clean_logger):
    """Test basic logger setup with default parameters"""
    log_stream = set_up_logger(temp_log_file)

    assert isinstance(log_stream, StringIO)
    assert os.path.exists(temp_log_file)


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_levels(temp_log_file, clean_logger, level):
    """Test if logger respects different log levels"""
    log_stream = set_up_logger(temp_log_file, logs_level=level)
    logger = logging.getLogger()
    test_message = "test message"

    assert logger.level == LOG_LEVELS[level]

    # Write a test message
    logger.log(LOG_LEVELS[level], test_message)

    # Check if message appears in the stream
    assert test_message in log_stream.getvalue()


def test_custom_format(temp_log_file, clean_logger):
    """Test if logger uses custom format correctly"""
    custom_format = "%(levelname)s: %(message)s"
    custom_date = "%Y-%m-%d"
    test_message = "test message"

    log_stream = set_up_logger(
        temp_log_file, log_format=custom_format, date_format=custom_date
    )

    logging.info(test_message)
    assert f"INFO: {test_message}" in log_stream.getvalue()


def test_invalid_file_path():
    """Test if logger raises OSError for invalid file path"""
    with pytest.raises(OSError):
        set_up_logger("/invalid/path/test.log")


def test_multiple_handlers(temp_log_file, clean_logger):
    """Test if all handlers are properly set up"""
    log_stream = set_up_logger(temp_log_file)
    logger = logging.getLogger()

    # Should have 3 handlers: file, console, and StringIO
    assert len(logger.handlers) == 3

    # Verify handler types
    handler_types = [type(h) for h in logger.handlers]
    assert logging.FileHandler in handler_types
    assert logging.StreamHandler in handler_types


def test_log_message_propagation(temp_log_file, clean_logger):
    """Test if messages propagate to both file and stream"""
    log_stream = set_up_logger(temp_log_file)
    test_message = "test propagation message"

    logging.info(test_message)

    # Check StringIO stream
    assert test_message in log_stream.getvalue()

    # Check file content
    with open(temp_log_file, "r") as f:
        assert test_message in f.read()
