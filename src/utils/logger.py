""" This modules has all the functions and classes to handle the logging. """

import logging
from logging import Logger
from typing import Literal

# Credits:
# alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.formats = {
            logging.DEBUG:
                self.fmt[0].replace("::reset", self.reset).replace("::color", self.grey),
            logging.INFO:
                self.fmt[0].replace("::reset", self.reset).replace("::color", self.blue),
            logging.WARNING:
                self.fmt[0].replace("::reset", self.reset).replace("::color", self.yellow),
            logging.ERROR:
                self.fmt[0].replace("::reset", self.reset).replace("::color", self.red),
            logging.CRITICAL:
                self.fmt[0].replace("::reset", self.reset).replace("::color", self.bold_red),
        }

    def format(self, record):
        log_fmt = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def set_logger_config_and_return(
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        no_color: bool
    ) -> Logger:
    """
    Sets the configuration for the logger and returns the root logger.

    Args:
        level (str): The log level to use.
            One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Returns:
        Logger: The root logger.
    """

    fmt = '%(asctime)s - ::color%(levelname)-8s::reset - %(name)s: %(message)s', # pylint: disable=trailing-comma-tuple
    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        format=fmt[0].replace("::reset", "").replace("::color", ""),
    )
    logger_temp = logging.getLogger('m')
    logger_temp.setLevel(level)
    if not no_color:
        logger_temp.propagate = False
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(CustomFormatter(fmt))
        logger_temp.addHandler(stdout_handler)
    return logger_temp
