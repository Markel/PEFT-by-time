""" Entry point for the project. Do -h for help. """

import os
import wandb

from .utils.torchtraining import full_training
from .dataset.dataset_downloader import download_dataset
from .utils.torchfuncs import get_device
from .utils.logger import set_logger_config_and_return
from .models.peft_convert import convert_to_peft

from .models.orig_model import download_model
from .utils.arguments import Args, parse_args, load_dotenv

ARGS: Args = parse_args()
DEVICE = get_device()

def main() -> None:
    """ Start executing the project """
    logger = set_logger_config_and_return(ARGS.debug, ARGS.no_color)
    logger.info("Arguments parsed and logger iniciated successfully. Welcome to the program.")
    logger.debug("Arguments: %s", ARGS)
    logger.info("Device: %s", DEVICE)

    logger.debug("Loading environment variables from .env file")
    try:
        load_dotenv()
        logger.info("Environment variables loaded successfully")
    except Exception as e: # pylint: disable=broad-except
        logger.error("Error loading environment variables: %s", e)

    model, tokenizer = download_model(ARGS.model, DEVICE)
    model   = convert_to_peft(model, ARGS)
    dataset = download_dataset(ARGS.dataset, tokenizer, DEVICE)

    os.environ["WANDB_SILENT"] = "true"
    if "WANDB_API_KEY" not in os.environ:
        logger.critical("WANDB_API_KEY not found in environment variables. Please set it up.")
    wandb.login(anonymous="never", key=os.environ["WANDB_API_KEY"], verify=True, force=True)
    logger.info("Logged in to Weights and Biases successfully")

    full_training(model, tokenizer, dataset, ARGS, DEVICE)
