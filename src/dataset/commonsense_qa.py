"""
This is a multiple-choice question answering dataset.

https://huggingface.co/datasets/tau/commonsense_qa
"""

import logging
from copy import copy
from typing import cast

import torch
from datasets import DatasetDict, load_dataset
from torch import Tensor, device
from torchmetrics import MetricCollection
from torchmetrics.classification import (BinaryAccuracy, BinaryF1Score,
                                         BinaryPrecision, BinaryRecall)
from transformers import T5TokenizerFast

from .base_dataset import BaseDataset

logger = logging.getLogger("m.dataset.commonsense_qa")

class CommonSenseQA(BaseDataset):
    """
    This class is the dataset for the commonsense_qa dataset.
    """

    def __init__(self, tokenizer: T5TokenizerFast, device_t: device, dev_size: float=0.05) -> None:
        super().__init__("tau/commonsense_qa")
        local_dir = "./downloads/datasets/tau/commonsense_qa"
        dataset_dict = load_dataset(local_dir)
        dataset_dict = cast(DatasetDict, dataset_dict)
        logger.debug("DatasetDict loaded successfully.")

        self.tokenizer = tokenizer

        # The dataset has no answers in the test set, so we will use the validation set for testing
        # and split the training set into training and validation sets.
        dataset_dict["test"] = dataset_dict["validation"]
        train_and_dev = dataset_dict["train"].train_test_split(test_size=dev_size, seed=42)
        dataset_dict["train"] = train_and_dev["train"]
        dataset_dict["validation"] = train_and_dev["test"]

        def get_correct_label_text(element) -> str:
            key: str = element["answerKey"]
            number: int = element["choices"]["label"].index(key)
            return "The correct answer for the question is: " + key + \
                   " - " + element["choices"]["text"][number]

        def get_correct_context_text(element) -> str:
            zip_list = list(zip(element["choices"]["label"], element["choices"]["text"]))
            return "The question is: " + element["question"] + \
                "\nThe possible answers are: \n" + \
                zip_list[0][0] + " - " + zip_list[0][1] + "\n" + \
                zip_list[1][0] + " - " + zip_list[1][1] + "\n" + \
                zip_list[2][0] + " - " + zip_list[2][1] + "\n" + \
                zip_list[3][0] + " - " + zip_list[3][1]

        dataset_dict = dataset_dict.map(
            lambda x: {"token_labels": tokenizer(
                                            get_correct_label_text(x),
                                            max_length=22,
                                            padding='max_length',
                                            truncation = True,
                                       )["input_ids"],
                       "labels": 1}, batched=False # Future work: batch
        )

        logger.debug("Labels converted successfully.")

        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(get_correct_context_text(x),
                                max_length=105,
                                padding="max_length",
                                truncation=True,
                                ), batched=False
        )

        logger.debug("DatasetDict converted successfully.")
        self.train = dataset_dict["train"].with_format("torch", device=device_t)
        self.dev = dataset_dict["validation"].with_format("torch", device=device_t)
        self.test = dataset_dict["test"].with_format("torch", device=device_t)
        logger.info("Dataset commonsense_qa loaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    def get_eval_methods(self, device_t: device) -> MetricCollection:
        """
        This method will return an evaluate a combination that contain the corresponding
        metrics to evaluate the dataset.
        """
        metric_collection = MetricCollection([
            BinaryAccuracy(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score()
        ]).to(device_t)
        return metric_collection

    def pre_eval_func(self, batch, token_labels) -> Tensor:
        """
        This method will return the tensor to be used for evaluation given the outputs
        of the model.

        It will match the output with the label token to token.
        """
        c_labels = copy(token_labels)
        c_batch = batch.logits.argmax(-1).tolist()
        c_labels[c_labels == -100] = 0
        c_labels = c_labels.tolist()

        # pylint: disable=consider-using-enumerate
        # Remove trailing padding of each possible answers for later comparison.
        for i in range(len(c_labels)):
            while c_labels[i][-1] == 0:
                c_labels[i] = c_labels[i][:-1]

        # pylint: enable=consider-using-enumerate

        output_label = [1 if pred[:len(gold)] == gold else 0 \
                        for pred, gold in zip(c_batch, c_labels)]
        output_label = torch.tensor(output_label).to(batch.logits.get_device())
        print(output_label)
        return output_label

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
