import logging


def get_or_create_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    existing_handlers = [handler for handler in logger.handlers if handler.level == log_level]
    if not existing_handlers:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)

        formatter = logging.Formatter('%(levelname)s [%(name)s] - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger
