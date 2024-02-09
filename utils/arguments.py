import argparse
from dataclasses import dataclass
from typing import Literal, cast

type DebugLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

@dataclass
class Args:
    debug: DebugLevel

def parseArgs() -> Args:
    """
    Parses the command line arguments and returns them.

    Returns:
        Args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="A program to test the efficiency of different PEFT methods through time.")

    parser.add_argument("--debug", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="The log level to use.")
    
    arguments = parser.parse_args()
    return cast(Args, arguments)