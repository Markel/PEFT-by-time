"""
This module serves for making the operations necessary to convert models to their
PEFT versions.

Available functions:
- convert_to_lora: Converts a model to a LoRA model.
"""

import logging
from typing import Union, cast

import adapters
from peft import PrefixTuningConfig, TaskType # type: ignore
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora.config import LoraConfig
from peft.tuners.vera.config import VeraConfig
from transformers import T5ForConditionalGeneration

from ..utils.arguments import Args
from ..utils.torchfuncs import get_device

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
    elif args.method == "prefix":
        logger.debug("Method selected: Prefix Tuning. Proceeding to convert the model.")
        if (isinstance(args.num_virtual_tokens, type(None)) or
            isinstance(args.prefix_projection, type(None))):
            logger.critical("Method Prefix Tuning requires num_virtual_tokens,\
                            and prefix_projection. Parser should have failed.")
            raise ValueError("Method Prefix Tuning requires num_virtual_tokens, \
                            and prefix_projection.")

        peft_model = convert_to_prefix_tuning(model,
                                             args.num_virtual_tokens, args.encoder_hidden,
                                             args.prefix_projection)
    elif args.method == "adapters":
        logger.debug("Method selected: Parallel Adapters. Proceeding to convert the model.")
        if isinstance(args.reduction_factor, type(None)):
            logger.critical("Method Parallel Adapters requires reduction_factor.\
                            Parser should have failed.")
            raise ValueError("Method Parallel Adapters requires reduction_factor.")
        peft_model = add_parallel_adapters(model, args.reduction_factor)
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
    logger.error("VeRA is not supported in the current version of the MAC counter.")
    return peft_model

def convert_to_prefix_tuning(model: T5ForConditionalGeneration,
                             num_virtual_tokens: int,
                             encoder_hidden: Union[int, None],
                             prefix_projection: bool,
                             ) -> PeftModel:
    """
    Converts a given T5 model to a PeftModel with Prefix Tuning configuration.
    Acodring to the original paper the prefix length: 200 for summarization, 10 for table-to-text

    Args:
        model (T5ForConditionalGeneration): The T5 model to convert.
        num_virtual_tokens (int): The number of virtual tokens to use in the prefix tuning.
        encoder_hidden (int|None): The number of hidden units in the encoder.
        prefix_projection (bool): If the prefix projection should be used.

    Returns:
        PeftModel: The converted PeftModel with Prefix Tuning configuration.
    """
    encoder_dict = {}
    if encoder_hidden is not None:
        encoder_dict["encoder_hidden_size"] = encoder_hidden

    config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=num_virtual_tokens,
        prefix_projection=prefix_projection,
        **encoder_dict
    )
    logger.debug("Prefix Tuning configuration created successfully.")
    peft_model = get_peft_model(model, config)
    peft_model = cast(PeftModel, peft_model)
    logger.info("New model's (Prefix Tuning) trainable parameters: %s. Total: %s (%s%s).",
                peft_model.get_nb_trainable_parameters()[0],
                peft_model.get_nb_trainable_parameters()[1],
                round(100 * peft_model.get_nb_trainable_parameters()[0]
                      / peft_model.get_nb_trainable_parameters()[1], 5),
                "%")
    return peft_model

def add_parallel_adapters(model: T5ForConditionalGeneration,
                          reduction_factor: float,
                          )-> PeftModel:
    """
    Adds parallel adapters to the given model.

    Args:
        model (T5ForConditionalGeneration): The model to add the adapters to.
        reduction_factor (float): The reduction factor to use in the adapters. It's float
        because the adapters library uses floats, but integers are expected. Default is 2.

    Returns:
        PeftModel: A casted model with the parallel adapters, in reality T5AdapterModel.
    """
    logger.debug("Converting model to adapters type.")
    num_orig = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapters.init(model)
    peft_model = cast(adapters.T5AdapterModel, model) # type: ignore
    config = adapters.ParBnConfig(reduction_factor=reduction_factor)
    peft_model.add_adapter("parallel_adapter", config=config)
    peft_model.train_adapter(["parallel_adapter"])
    peft_model.set_active_adapters(["parallel_adapter"])
    peft_model.to(get_device()) # type: ignore
    num_new  = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info("New model's (Adapters) trainable parameters: %s. Total: %s (%s%s).",
                num_new, num_orig, round(100 * num_new / num_orig, 5), "%")
    peft_model = cast(PeftModel, peft_model)
    return peft_model
