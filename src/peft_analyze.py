""" Entry point for the project. Do -h for help. """

import logging
from logging import Logger
from typing import Literal

from .models.orig_model import download_model
from .utils.arguments import Args, parse_args

ARGS: Args = parse_args()

def set_logger_config_and_return(
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ) -> Logger:
    """
    Sets the configuration for the logger and returns the root logger.

    Args:
        level (str): The log level to use.
            One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Returns:
        Logger: The root logger.
    """

    logging.basicConfig(
        format='%(asctime)s - %(levelname)-8s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger_temp = logging.getLogger('m')
    logger_temp.setLevel(level)
    return logger_temp

def main():
    """ Start executing the project """
    logger = set_logger_config_and_return(ARGS.debug)
    logger.debug("Arguments parsed and logger iniciated successfully. Welcome to the program.")
    logger.debug("Arguments: %s", ARGS)
    model, tokenizer  = download_model(ARGS.model) # pylint: disable=unused-variable
