"""
This module contains the abstract class for all datasets, so we can do type checking effectively.
"""

import logging
logger = logging.getLogger("dataset.base_dataset")

def test_debug_function():
    """ A function to test the logger and the classes. """
    logger.info("foo() called")
    return "foo"

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
