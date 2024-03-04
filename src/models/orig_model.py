"""
This module serves to download the model from Huggingface and return the tokenizer and the model.
"""

import logging
import os
from typing import cast
from huggingface_hub import snapshot_download
from torch import device
from transformers import T5TokenizerFast, T5ForConditionalGeneration

logger = logging.getLogger("m.models.orig_model")

def download_model(model: str, init_device: device) -> tuple[T5ForConditionalGeneration, T5TokenizerFast]:
    """
    Given the name of a model it download the corresponding model from Huggingface.
    If already downloaded it just loads the model.
    It downloads it to the downloads folder, and then loads it from there.

    Args:
        model (str): The name of the model to download.
        device (torch.device): Device save the model to.

    Returns:
        tuple[T5ForConditionalGeneration, T5TokenizerFast]: The model and the tokenizer.
        It's set to T5, but Seq2Seq models should work as well (not guaranteed).
    """
    model_repo: str
    local_dir = "./downloads/models/"+model
    # Would be the best match .. case example, but 3.9 compatibility is necessary :/
    if model == "small":
        model_repo = "google-t5/t5-small"
    elif model == "base":
        model_repo = "google-t5/t5-base"
    elif model == "large":
        model_repo = "google-t5/t5-large"
    elif model == "xl":
        model_repo = "google-t5/t5-xl"
    elif model == "xxl":
        model_repo = "google-t5/t5-xxl"
    else:
        model_repo = model

    model_dir = local_dir
    if not os.path.isdir(local_dir):
        logger.info("Downloading the model from the internet.")
        model_dir = snapshot_download(model_repo, local_dir=local_dir,
                                ignore_patterns=["*.msgpack", "*.h5", "rust_model.ot", "*.onnx"])
        logger.debug("Model downloaded successfully.")
    logger.debug("Model identified successfully. Proceeding to load the tokenizer and model.")

    tokenizer: T5TokenizerFast = T5TokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model_hug  = T5ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True, device_map=init_device)
    model_hug  = cast(T5ForConditionalGeneration, model_hug)
    logger.debug("Model and tokenizer loaded successfully.")

    return (model_hug, tokenizer)
