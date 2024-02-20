""" Entry point for the project. Do -h for help. """

from gc import disable
from logging import Logger
import logging

from pyparsing import disable_diag
from typing import Literal
from .utils.arguments import parse_args, Args
from .models.orig_model import download_model

ARGS: Args = parse_args()

def set_logger_config_and_return(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]) -> Logger:
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
    logger.debug(f"Arguments: {ARGS}")
    model  = download_model(ARGS.model)