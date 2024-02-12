""" This module contains the function and class type to parse the command line arguments. """

import argparse
from dataclasses import dataclass
from typing import Literal, cast

type DebugLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

@dataclass
class Args(argparse.Namespace):
    """
    A holder class to sustitute the argparse.Namespace class,
    but including a property for each entry parameter.
    """

    # The level for the logging https://docs.python.org/3/library/logging.html#logging-levels
    debug: DebugLevel

def parse_args() -> Args:
    """
    Parses the command line arguments and returns them.

    Returns:
        Args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="A program to test the efficiency of different PEFT methods through time."
    )

    parser.add_argument("--debug", type=str, default="INFO", help="The log level to use.",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )

    arguments = parser.parse_args()
    return cast(Args, arguments)
