""" Entry point for the project. Do -h for help. """

from logging import Logger
import logging
from utils.arguments import parse_args, DebugLevel

def set_logger_config_and_return(level: DebugLevel) -> Logger:
    """
    Sets the configuration for the logger and returns the root logger.

    Args:
        level (DebugLevel): The log level to use.
            One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Returns:
        Logger: The root logger.
    """

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)-8s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('root')

if __name__ == "__main__":
    ARGS = parse_args()
    logger = set_logger_config_and_return(ARGS.debug)
    logger.debug("Arguments parsed and logger iniciated successfully. Welcome to the program.")
