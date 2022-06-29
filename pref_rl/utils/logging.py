import logging


# TODO: Modify to get_or_create_logger
def create_logger(location):
    logger = logging.getLogger(location)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s [%(name)s] - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger
