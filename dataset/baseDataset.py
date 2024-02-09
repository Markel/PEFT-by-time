import logging

def foo():
    logger = logging.getLogger("dataset.baseDataset")
    logger.info("foo() called")
    return "foo"

if __name__ == "__main__":
    logger = logging.getLogger("dataset.baseDataset")
    logger.critical("Module not meant to be run as a script. Exiting.")