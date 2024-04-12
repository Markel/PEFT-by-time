"""

"""

import logging
from typing import cast

from datasets import DatasetDict, load_dataset
from torch import Tensor, device
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)
from transformers import T5TokenizerFast

from .base_dataset import BaseDataset

logger = logging.getLogger("m.dataset.ag_news")

class AGNews(BaseDataset):
    """
    This class is the dataset for the AGNews dataset.
    """

    tokenized_results: list[list[int]] = []
    """
    Embeddings per class:
    [37, 2859, 13, 8, 1108, 19, 10, 1150, 1, 0, 0],
    [37, 2859, 13, 8, 1108, 19, 10, 5716, 1, 0, 0],
    [37, 2859, 13, 8, 1108, 19, 10, 1769, 1, 0, 0],
    [37, 2859, 13, 8, 1108, 19, 10, 16021, 87, 9542, 1]
    """

    def __init__(self, tokenizer: T5TokenizerFast, device_t: device, dev_size: float=0.05) -> None:
        super().__init__("ag_news")
        local_dir = "./downloads/datasets/ag_news"
        dataset_dict = load_dataset(local_dir)
        dataset_dict = cast(DatasetDict, dataset_dict)
        logger.debug("DatasetDict loaded successfully.")

        longest = tokenizer.encode("The topic of the article is: Sci/Tech")
        self.tokenized_results.append(tokenizer.encode("The topic of the article is: World",
                                      padding="max_length", max_length=len(longest)))
        self.tokenized_results.append(tokenizer.encode("The topic of the article is: Sports",
                                      padding="max_length", max_length=len(longest)))
        self.tokenized_results.append(tokenizer.encode("The topic of the article is: Business",
                                      padding="max_length", max_length=len(longest)))
        self.tokenized_results.append(longest)
      
        dataset_dict = dataset_dict.map(
            lambda x: {"token_labels": self.tokenized_results[x["label"]]}
        )
        dataset_dict = dataset_dict.rename_column("label", "labels")
        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(x["text"], max_length=441, padding="max_length", truncation=True,
                                return_tensors="pt"), batched=True
        )
        logger.debug("DatasetDict converted successfully.")

        print(dataset_dict)
        train_and_dev = dataset_dict["train"].train_test_split(test_size=dev_size, seed=42)
        self.train = train_and_dev["train"].with_format("torch", device=device_t)
        self.dev = train_and_dev["test"].with_format("torch", device=device_t)
        self.test = dataset_dict["test"].with_format("torch", device=device_t)
        logger.info("Dataset ag_news loaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    def get_eval_methods(self, device_t: device) -> MetricCollection:
        """
        This method will return an evaluate a combination that contain the corresponding
        metrics to evaluate the dataset.
        """
        raise NotImplementedError

    def pre_eval_func(self, batch) -> Tensor:
        """
        This method will return the tensor to be used for evaluation given the outputs
        of the model.

        Negative answers are considered for all cases where it doens't match the
        positive label. However, when computing the loss, these are independent classes.
        """
        raise NotImplementedError

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
