"""
This module serves to download the model from Huggingface and return the tokenizer and the model.
"""

import logging
import os
from typing import cast
from huggingface_hub import snapshot_download
from transformers import T5Model, T5TokenizerFast

logger = logging.getLogger("m.models.orig_model")

def download_model(model: str) -> tuple[T5Model, T5TokenizerFast]:
    """
    Given the name of a model it download the corresponding model from Huggingface.
    If already downloaded it just loads the model.
    It downloads it to the downloads folder, and then loads it from there.

    Args:
        model (str): The name of the model to download.

    Returns:
        tuple[T5Model, T5TokenizerFast]: The model and the tokenizer.
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
    model_hug  = T5Model.from_pretrained(model_dir, local_files_only=True)
    model_hug  = cast(T5Model, model_hug)
    logger.debug("Model and tokenizer loaded successfully.")

    return (model_hug, tokenizer)
