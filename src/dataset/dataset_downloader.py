"""
Defines a function to download the correct dataset.

"""

from typing import Literal

from transformers import T5TokenizerFast
from torch import device

from .ag_news import AGNews

from .base_dataset import BaseDataset
from .tweet_eval import TweetEvalHate


def download_dataset(dataset_name: Literal["tweet_eval", "ag_news"],
                     tokenizer: T5TokenizerFast,
                     device_t: device
                    ) -> BaseDataset:
    """
    Downloads the specified dataset.

    Args:
        dataset_name (Literal["tweet_eval", "ag_news"]): The name of the dataset to download.

    Returns:
        BaseDataset: The downloaded dataset.

    Raises:
        ValueError: If the specified dataset is not found.
    """
    if dataset_name == "tweet_eval":
        return TweetEvalHate(tokenizer, device_t)
    if dataset_name == "ag_news":
        return AGNews(tokenizer, device_t)
    raise ValueError(f"Dataset {dataset_name} not found.")
