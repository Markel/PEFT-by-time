from dataset.baseDataset import foo
from utils.arguments import Args, parseArgs, DebugLevel
import logging
from logging import Logger

def setLoggerConfigAndReturn(level: DebugLevel) -> Logger:
    """
    Sets the configuration for the logger and returns the root logger.

    Args:
        level (DebugLevel): The log level to use. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

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
    ARGS = parseArgs()
    logger = setLoggerConfigAndReturn(ARGS.debug)
    logger.debug("Arguments parsed and logger iniciated successfully. Welcome to the program.")
    foo()