"""
This module serves for making the operations necessary to convert models to their
PEFT versions.

Available functions:
- convert_to_lora: Converts a model to a LoRA model.
"""

import logging
from typing import cast

from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora.config import LoraConfig
from peft.tuners.vera.config import VeraConfig
from transformers import T5ForConditionalGeneration
from ..utils.arguments import Args

logger = logging.getLogger("m.models.peft_convert")

def convert_to_peft(model: T5ForConditionalGeneration, args: Args) -> PeftModel:
    """
    Converts the given T5 model to a PeftModel based on the specified method.

    Args:
        model (T5ForConditionalGeneration): The T5 model to be converted.
        args (Args): The inline arguments of the program.

    Returns:
        PeftModel: The converted PeftModel.

    Raises:
        ValueError: If the args are not correctly passed.
    """
    if args.method == "LoRA":
        logger.debug("Method selected: LoRA. Proceeding to convert the model.")
        if (isinstance(args.rank, type(None)) or isinstance(args.alpha, type(None)) or
            isinstance(args.dropout, type(None)) or isinstance(args.target_modules, type(None))):
            logger.critical("Method LoRA requires rank, alpha, dropout and target modules. \
                            Parser should have failed.")
            raise ValueError("Method LoRA requires rank, alpha, dropout and target modules.")

        peft_model = convert_to_lora(model,
                                     args.rank, args.alpha, args.dropout, args.target_modules)
    elif args.method == "VeRA":
        logger.debug("Method selected: VeRA. Proceeding to convert the model.")
        if (isinstance(args.rank, type(None)) or isinstance(args.d_initial, type(None)) or
            isinstance(args.dropout, type(None)) or isinstance(args.target_modules, type(None))):
            logger.critical("Method VeRA requires rank, d_initial, dropout and target modules. \
                            Parser should have failed.")
            raise ValueError("Method VeRA requires rank, d_initial, dropout and target modules.")

        peft_model = convert_to_vera(model,
                                     args.rank, args.d_initial, args.dropout, args.target_modules)
    elif args.method == "FT":
        logger.debug("Method selected: Full fine-tuning. Proceeding to \"convert\" the model.")
        peft_model = dummy_fft_convert(model)
    else:
        logger.critical("Method %s not recognized, parser should have failed.", args.method)
        raise ValueError("Method not recognized.")
    return peft_model

def convert_to_lora(model: T5ForConditionalGeneration,
                    rank: int, alpha: int,
                    dropout: float,
                    target_modules: list[str],
                    ) -> PeftModel:
    """
    Converts a given T5Model to a PeftModel with LoRA configuration.

    Args:
        model (T5ForConditionalGeneration): The T5Model to convert.
        rank (int): The rank parameter for LoRA.
        alpha (int): The alpha parameter for LoRA.
        dropout (float): The dropout rate for LoRA.
        target_modules (list[str]): The list of target modules for LoRA.
        
    Returns:
        PeftModel: The converted PeftModel with LoRA configuration.

    Notes:
        - The bias is set to "lora_only".
        - The modules to save are set to ["decode_head"].
        - The function should work with non T5 models, but it's not guaranteed.
    """

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="lora_only",
        modules_to_save=["decode_head"],
    )
    logger.debug("LoRA configuration created successfully.")
    peft_model = get_peft_model(model, config)
    peft_model = cast(PeftModel, peft_model)
    logger.info("New model's (LoRA) trainable parameters: %s. Total: %s (%s%s).",
                peft_model.get_nb_trainable_parameters()[0],
                peft_model.get_nb_trainable_parameters()[1],
                round(100 * peft_model.get_nb_trainable_parameters()[0]
                      / peft_model.get_nb_trainable_parameters()[1], 5),
                "%")
    return peft_model

def dummy_fft_convert(model: T5ForConditionalGeneration) -> PeftModel:
    """
    Dummy function to convert a model to a "Full fine-tuning" model.

    Args:
        model (T5ForConditionalGeneration): The model to convert.

    Returns:
        PeftModel: The converted model.

    Notes:
        - This function is a placeholder for future implementations.
    """
    logger.debug("Dummy function to convert to Full fine-tuning model.")
    # Count number of parameters
    logger.info("New model's (Full fine-tuning) trainable parameters: %s. Total: %s (%s%s).",
                model.num_parameters(only_trainable=True),
                model.num_parameters(),
                100,
                "%")
    return cast(PeftModel, model)

def convert_to_vera(model: T5ForConditionalGeneration,
                    rank: int, d_initial: float,
                    dropout: float,
                    target_modules: list[str],
                    ) -> PeftModel:
    """
    Converts a given T5 model to a VeRA model.

    Args:
        model (T5ForConditionalGeneration): The T5 model to convert.
        rank (int): The rank of the VeRA model.
        d_initial (float): The initial value of the d parameter in VeRA.
        dropout (float): The dropout rate for VeRA.
        target_modules (list[str]): The list of target modules for VeRA.

    Returns:
        PeftModel: The converted VeRA model.

    Notes:
        - As of April 2024 peft needs to be installed from the source code for VeRA support.
        - The function should work with non T5 models, but it's not guaranteed, specially
        taking into account the limitations of the current implementation.
        - The bias is set to "vera_only".
        - The modules to save are set to ["decode_head"].
        
    """
    
    config = VeraConfig(
        r=rank,
        d_initial=d_initial,
        target_modules=target_modules,
        vera_dropout=dropout,
        bias="vera_only",
        modules_to_save=["decode_head"],
    )
    logger.debug("VeRA configuration created successfully.")
    peft_model = get_peft_model(model, config)
    peft_model = cast(PeftModel, peft_model)
    logger.info("New model's (VeRA) trainable parameters: %s. Total: %s (%s%s).",
                peft_model.get_nb_trainable_parameters()[0],
                peft_model.get_nb_trainable_parameters()[1],
                round(100 * peft_model.get_nb_trainable_parameters()[0]
                      / peft_model.get_nb_trainable_parameters()[1], 5),
                "%")
    return peft_model
