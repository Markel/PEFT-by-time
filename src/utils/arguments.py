""" This module contains the function and class type to parse the command line arguments. """

import argparse
import os
from dataclasses import dataclass
from typing import Literal, Optional, cast

@dataclass
class Args(argparse.Namespace):
    """
    A holder class to sustitute the argparse.Namespace class,
    but including a property for each entry parameter.
    """

    # The batch size to use in the training.
    batch_size: int
    # Name of the dataset to use.
    dataset: Literal["tweet_eval"]
    # The level for the logging https://docs.python.org/3/library/logging.html#logging-levels
    debug: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    # Number of epochs to train the model. Default to 10.
    epochs: int
    # Number of steps to evaluate the model. Default to -1, meaning evaluation after every epoch.
    eval_every: int
    # The model to use. "small","base","large","xl","xxl" should be defaulted to T5v1.1 versions.
    model: str
    # The learning rate to use in the optimizer.
    learning_rate: float
    # The model to use.
    model: str
    # The method to use.
    method: Literal["LoRA"]
    # If the output should be colored.
    no_color: bool

    ##* W&B parameters
    # Custom name for the experiment.
    experiment_name: Optional[str] = None
    # The project to log the experiment.
    project: str = "peft-by-time"

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
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="The batch size to use in the training.")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="The dataset to use.",
                        choices=["tweet_eval"])
    parser.add_argument("--debug", type=str, default="INFO", help="The log level to use.",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("-e", "--epochs", type=int, default=10,
                        help="Number of epochs to train the model. Default to 10.")
    parser.add_argument("-ee", "--eval_every", type=int, default=-1,
                        help="Number of steps to evaluate the model. Default to -1, meaning \
                              evaluation after every epoch.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="The learning rate to use in the optimizer.")
    parser.add_argument("-m", "--model", type=str, default="base",
                        help="The model to use. \"small\", \"base\", \"large\", \"xl\", \"xxl\" \
                              default to the corresponding T5v1.1 versions. Other models should \
                              be passed in huggingface repo format.")
    parser.add_argument("-nc", "--no_color", action="store_true",
                        help="Disable color in the output of the logger.")

    #** W&B
    parser.add_argument("-Wen", "--experiment_name", type=str,
                        help="W&B custom name for the experiment.")
    parser.add_argument("-Wp", "--project", type=str,
                        default="peft-by-time", help="The W&B project to log the experiment.")
    parser.add_argument("-Wt", "--run_type", type=str, default="run_test",
                        help="Extra parameter for informational value for later grouping. \
                              Defaults to \"run_test\".")

    #** METHODS
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

def load_dotenv():
    """
    Loads the environment variables from the .env file.

    Reads each line of the .env file, extracts the key-value pairs,
    and sets them as environment variables using os.environ.

    Note: The .env file should be present in the same directory as this script.

    Example:
    If the .env file contains the following:
    ```
    API_KEY=123456789
    ```
    Then after calling load_dotenv(), the environment variables API_KEY and DEBUG
    will be set with the respective values.

    Raises:
    FileNotFoundError: If the .env file is not found.

    ValueError: If the .env file contains invalid key-value pairs.

    """
    with open('.env', encoding="utf-8") as f:
        for line in f:
            key, value = line.strip().split('=')
            os.environ[key] = value
