import logging

from pref_rl.utils.logging import get_or_create_logger


def test_creates_logger():
    logger = get_or_create_logger("TestLogger")
    assert isinstance(logger, logging.Logger)


def test_creates_stream_handler():
    logger = get_or_create_logger("TestLogger")
    assert isinstance(logger.handlers[0], logging.StreamHandler)


def test_sets_correct_name():
    logger_name = "TestLogger"
    logger = get_or_create_logger(logger_name)
    assert logger.name == logger_name


def test_sets_correct_log_level():
    log_level = logging.DEBUG
    logger = get_or_create_logger("TestLogger-log-level", log_level=log_level)
    assert logger.handlers[0].level == log_level


def test_no_duplicate_handlers_are_added_to_logger():
    _ = get_or_create_logger("TestLogger-duplicate", log_level=logging.DEBUG)
    logger = get_or_create_logger("TestLogger-duplicate", log_level=logging.DEBUG)
    assert len(logger.handlers) == 1


def test_handlers_with_different_log_levels_are_added():
    _ = get_or_create_logger("TestLogger-different-levels", log_level=logging.INFO)
    logger = get_or_create_logger("TestLogger-different-levels", log_level=logging.DEBUG)
    assert len(logger.handlers) == 2
