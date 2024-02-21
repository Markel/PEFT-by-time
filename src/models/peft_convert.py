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
from transformers import T5Model

logger = logging.getLogger("m.models.peft_convert")

def convert_to_lora(model: T5Model,
                    rank: int, alpha: int,
                    dropout: float,
                    target_modules: list[str],
                    ) -> PeftModel:
    """
    Converts a given T5Model to a PeftModel with LoRA configuration.

    Args:
        model (T5Model): The T5Model to convert.
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
    logger.info("New model's (LoRA) trainable parameters: %s. Previous: %s (%s%s).",
                peft_model.get_nb_trainable_parameters()[0],
                peft_model.get_nb_trainable_parameters()[1],
                round(100 * peft_model.get_nb_trainable_parameters()[0]
                      / peft_model.get_nb_trainable_parameters()[1], 5),
                "%")
    return peft_model
