"""
This module contains the abstract class for all datasets, so we can do type checking effectively.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("dataset.base_dataset")

class BaseDataset(ABC):
    """
    This is an abstract class for having typing available for the different dataset.
    SHOULD NOT be instanciated directly, higher hierarchy classes should be used.
    """
    train, dev, test = None, None, None

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def eval(self):
        """
        TODO: Add docstring
        """

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
