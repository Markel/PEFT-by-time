"""
Imports the TweetEval dataset, in specific the hate dataset.

More information of the dataset can be found in the following link:
https://huggingface.co/datasets/tweet_eval/

This dataset is a Multiclass classification dataset and the following metrics are used:
Accuracy, Recall, Precision, F1 Score.
"""

import logging
from typing import cast

from datasets import DatasetDict, load_dataset
from torch import Tensor, device
import torch
from torch.nn import MSELoss
from torch.nn.modules import Module
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy, MulticlassF1Score,
                                         MulticlassPrecision, MulticlassRecall)
from transformers import T5TokenizerFast

from .base_dataset import BaseDataset

logger = logging.getLogger("m.dataset.tweet_eval")

class TweetEvalHate(BaseDataset):
    """
    This class is the dataset for the TweetEval dataset.
    """

    pos: list[int]
    neg: list[int]

    def __init__(self, tokenizer: T5TokenizerFast, device_t: device) -> None:
        super().__init__("tweet_eval")
        local_dir = "./downloads/datasets/tweet_eval"
        dataset_dict = load_dataset(local_dir, "hate")
        dataset_dict = cast(DatasetDict, dataset_dict)
        logger.debug("DatasetDict loaded successfully.")

        self.pos = tokenizer.encode('positive')
        self.neg = tokenizer.encode('negative')
        dataset_dict = dataset_dict.map(
            lambda x: {"token_labels": self.pos if x["label"] == 1 else self.neg}
        )
        dataset_dict = dataset_dict.rename_column("label", "labels")
        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(x["text"], padding=True, truncation=True, return_tensors="pt"
                                ), batched=True
        )
        logger.debug("DatasetDict converted successfully.")

        self.train = dataset_dict["train"].with_format("torch", device=device_t)
        self.dev = dataset_dict["validation"].with_format("torch", device=device_t)
        self.test = dataset_dict["test"].with_format("torch", device=device_t)
        logger.info("Dataset tweet_eval loaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    def get_eval_methods(self, device_t: device) -> MetricCollection:
        # I use multiclass to leave out as 0 the starting non-sensical text.
        # Posdata: They should be the same because they are micro.
        metric_collection = MetricCollection([
            MulticlassAccuracy(3, average="micro"),
            MulticlassPrecision(3, average="micro"),
            MulticlassRecall(3, average="micro"),
            MulticlassF1Score(3, average="micro")
        ]).to(device_t)
        return metric_collection

    def pre_eval_func(self, batch) -> Tensor:
        output_label = batch.logits.argmax(-1)
        output_label = output_label.tolist()
        output_label = [1 if x == self.pos else 0 if x == self.neg else -1 for x in output_label]
        output_label = torch.tensor(output_label).to(batch.logits.get_device())
        return output_label

    def get_loss_function(self) -> Module:
        return MSELoss()

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
