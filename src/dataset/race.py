"""
This dataset is a collection of .

Special thank to Kristian Vachev:
https://github.com/KristiyanVachev/Leaf-Question-Generation
"""

import logging
from typing import cast

from datasets import DatasetDict, load_dataset
from torch import device
import torch
from torchmetrics import MetricCollection
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore # pylint:disable=unused-import

from transformers import T5TokenizerFast

from .base_dataset import BaseDataset

logger = logging.getLogger("m.dataset.race")

class Race(BaseDataset):
    """
    This class is the dataset for the Race dataset.
    """

    # This dataset requires decoding of text so the tokenizer needs to be saved
    tokenizer: T5TokenizerFast
    SEP_TOKEN: str = '<sep>'

    def __init__(self, tokenizer: T5TokenizerFast, device_t: device) -> None:
        super().__init__("ehovy/race")
        local_dir = "./downloads/datasets/ehovy/race"
        dataset_dict = load_dataset(local_dir, 'all')
        dataset_dict = cast(DatasetDict, dataset_dict)
        logger.debug("DatasetDict loaded successfully.")

        tokenizer.add_tokens(self.SEP_TOKEN)
        self.tokenizer = tokenizer
        options = ["A", "B", "C", "D"]

        def get_correct_context_text(element) -> str:
            number: int = options.index(element["answer"])
            # If we put the article at the end, that will be what's truncated
            return f'{element["options"][number]} {self.SEP_TOKEN} {element["question"]} {self.SEP_TOKEN} {element["article"]}' # pylint: disable=line-too-long

        def get_correct_label_text(element) -> str:
            numbers = [0, 1, 2, 3]
            numbers.remove(options.index(element["answer"]))
            return f'{element["options"][numbers[0]]} {self.SEP_TOKEN} {element["options"][numbers[1]]} {self.SEP_TOKEN} {element["options"][numbers[2]]}' # pylint: disable=line-too-long


        dataset_dict = dataset_dict.map(
            lambda x: {"token_labels": tokenizer(
                                            get_correct_label_text(x),
                                            max_length=221,
                                            padding='max_length',
                                            truncation = True,
                                            add_special_tokens=True,
                                       )["input_ids"],
                       "labels": get_correct_label_text(x)}, batched=False # Future work: batch
        )

        logger.debug("Labels converted successfully.")

        # The required context for no truncation is 1618. 511 is the maximum for T5
        # if we use 1618, the model will require more resources.
        context_max_length: int = 511
        if context_max_length > 511:
            logger.warning("Context max length is greater than 511.\
                           A lot of resources may be used.")

        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(get_correct_context_text(x),
                                max_length=context_max_length,
                                padding="max_length",
                                truncation=True,
                                add_special_tokens=True,
                                ), batched=False
        )

        logger.debug("DatasetDict converted successfully.")
        self.train = dataset_dict["train"].with_format("torch", device=device_t)
        self.dev = dataset_dict["validation"].with_format("torch", device=device_t)
        self.test = dataset_dict["test"].with_format("torch", device=device_t)
        logger.info("Dataset race loaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    def get_eval_methods(self, device_t: device) -> MetricCollection:
        """
        This method will return an evaluate a combination that contain the corresponding
        metrics to evaluate the dataset.
        """
        metric_collection = MetricCollection({
            "BLEU1": BLEUScore(n_gram=1, smooth=True),
            "BLEU2": BLEUScore(n_gram=2, smooth=True),
            "BLEU3": BLEUScore(n_gram=3, smooth=True),
            "BLEU4": BLEUScore(n_gram=4, smooth=True),
            #"ROGUE": ROUGEScore(), # I would actually use only ROGUE_L, but let's compute all
        }).to(device_t)
        return metric_collection

    def pre_eval_func(self, batch) -> list[str]:
        """
        This method will return the tensor to be used for evaluation given the outputs
        of the model.

        In this case the return vector will be the decoded input ids.
        """
        output_label = self.tokenizer.batch_decode(
            torch.argmax(batch.logits, dim=2).tolist(), skip_special_tokens=True
        )
        #output_label = torch.tensor(output_label).to(batch.logits.get_device())
        return output_label

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
