"""
Defines a function to download the correct dataset.

"""

from typing import Literal

from transformers import T5TokenizerFast
from torch import device

from .base_dataset import BaseDataset
from .tweet_eval import TweetEvalHate


def download_dataset(dataset_name: Literal["tweet_eval"],
                     tokenizer: T5TokenizerFast,
                     device_t: device
                    ) -> BaseDataset:
    """
    Downloads the specified dataset.

    Args:
        dataset_name (Literal["tweet_eval"]): The name of the dataset to download.

    Returns:
        BaseDataset: The downloaded dataset.

    Raises:
        ValueError: If the specified dataset is not found.
    """
    if dataset_name == "tweet_eval":
        return TweetEvalHate(tokenizer, device_t)
    raise ValueError(f"Dataset {dataset_name} not found.")
