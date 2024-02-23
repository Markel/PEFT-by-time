""" Entry point for the project. Do -h for help. """

from .dataset.dataset_downloader import download_dataset
from .utils.torchfuncs import get_device
from .utils.logger import set_logger_config_and_return
from .models.peft_convert import convert_to_peft

from .models.orig_model import download_model
from .utils.arguments import Args, parse_args

ARGS: Args = parse_args()
DEVICE = get_device()

def main() -> None:
    """ Start executing the project """
    logger = set_logger_config_and_return(ARGS.debug, ARGS.no_color)
    logger.info("Arguments parsed and logger iniciated successfully. Welcome to the program.")
    logger.debug("Arguments: %s", ARGS)
    logger.info("Device: %s", DEVICE)
    model, tokenizer = download_model(ARGS.model, DEVICE) # pylint: disable=unused-variable
    model   = convert_to_peft(model, ARGS)
    dataset = download_dataset(ARGS.dataset) # pylint: disable=unused-variable
