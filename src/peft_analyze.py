""" Entry point for the project. Do -h for help. """

from .utils.logger import set_logger_config_and_return
from .models.peft_convert import convert_to_lora

from .models.orig_model import download_model
from .utils.arguments import Args, parse_args

ARGS: Args = parse_args()

def main() -> None:
    """ Start executing the project """
    logger = set_logger_config_and_return(ARGS.debug, ARGS.no_color)
    logger.info("Arguments parsed and logger iniciated successfully. Welcome to the program.")
    logger.debug("Arguments: %s", ARGS)
    model, tokenizer  = download_model(ARGS.model) # pylint: disable=unused-variable

    #* Get PEFT-ize version of the model
    if ARGS.method == "LoRA":
        logger.debug("Method selected: LoRA. Proceeding to convert the model.")
        if (isinstance(ARGS.rank, type(None)) or isinstance(ARGS.alpha, type(None)) or
            isinstance(ARGS.dropout, type(None)) or isinstance(ARGS.target_modules, type(None))):
            logger.critical("Method LoRA requires rank, alpha, dropout and target modules. \
                            Parser should have failed.")
            return
        model = convert_to_lora(model, ARGS.rank, ARGS.alpha, ARGS.dropout, ARGS.target_modules)
    else:
        logger.critical("Method %s not recognized, parser should have failed.", ARGS.method)
        return
