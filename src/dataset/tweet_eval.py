"""
Imports the TweetEval dataset, in specific the hate dataset.

More information of the dataset can be found in the following link:
https://huggingface.co/datasets/tweet_eval/

This dataset is a binary classification dataset and the following metrics are used:
Accuracy, Recall, Precision, F1 Score.
"""

import logging
from typing import cast

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryRecall, BinaryAccuracy, BinaryPrecision, BinaryF1Score

from datasets import DatasetDict, load_dataset

from .base_dataset import BaseDataset

logger = logging.getLogger("m.dataset.tweet_eval")

class TweetEvalHate(BaseDataset):
    """
    This class is the dataset for the TweetEval dataset.
    """

    def __init__(self) -> None:
        super().__init__("tweet_eval")
        local_dir = "./downloads/datasets/tweet_eval"
        dataset_dict = load_dataset(local_dir, "hate")
        dataset_dict = cast(DatasetDict, dataset_dict)
        logger.debug("DatasetDict loaded successfully.")
        self.train = dataset_dict["train"]
        self.dev = dataset_dict["validation"]
        self.test = dataset_dict["test"]
        logger.info("Dataset tweet_eval loaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    def get_eval_methods(self) -> MetricCollection:
        metric_collection = MetricCollection([
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score()
        ])
        return metric_collection

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
