""" This module contains the function and class type to parse the command line arguments. """

import argparse
from dataclasses import dataclass
from typing import Literal, Optional, cast

@dataclass
class Args(argparse.Namespace):
    """
    A holder class to sustitute the argparse.Namespace class,
    but including a property for each entry parameter.
    """

    # The level for the logging https://docs.python.org/3/library/logging.html#logging-levels
    debug: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    # The model to use. "small","base","large","xl","xxl" should be defaulted to T5v1.1 versions.
    model: str
    # The method to use.
    method: Literal["LoRA"]
    # If the output should be colored.
    no_color: bool

    ##* LoRA parameters
    # The rank to set LoRA to.
    rank: Optional[int] = None
    # The alpha to set LoRA to.
    alpha: Optional[int] = None
    # The dropout to set LoRA to.
    dropout: Optional[float] = None
    # The modules to target. Default to ["query", "value"].
    target_modules: Optional[list[str]] = None


def parse_args() -> Args:
    """
    Parses the command line arguments and returns them.

    Returns:
        Args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="A program to test the efficiency of different PEFT methods through time."
    )
    subparsers = parser.add_subparsers(dest="method",
                                       help='Available methods. Use method\'s help to know the \
                                             specific parameter',
                                       required=True)

    parser.add_argument("--debug", type=str, default="INFO", help="The log level to use.",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("-m", "--model", type=str, default="base",
                        help="The model to use. \"small\", \"base\", \"large\", \"xl\", \"xxl\" \
                              default to the corresponding T5v1.1 versions. Other models should \
                              be passed in huggingface repo format.")
    parser.add_argument("-nc", "--no-color", action="store_true",
                        help="Disable color in the output of the logger.")

    #* Lora method
    parser_lora = subparsers.add_parser('LoRA', help='Low-Rank Adaptation')
    parser_lora.add_argument('-r', '--rank', type=int,
                             help='The rank to set LoRA to.', required=True)
    parser_lora.add_argument('-a', '--alpha', type=int,
                             help='The alpha to set LoRA to.', required=True)
    parser_lora.add_argument('-d', '--dropout', type=float,
                             help='The dropout to set LoRA to.', required=True)
    parser_lora.add_argument('-t', '--target_modules', type=str, nargs='+',
                             default=["q", "v"],
                             help='The modules to target. Default to ["q", "v"].')

    arguments = parser.parse_args()
    return cast(Args, arguments)
