"""
TweetEval-Hate is sourced from the SemEval 2019 Task 5, which focuses on the detection of hate
speech against immigrants and women in messages extracted from Twitter. The tweets extracted from
Twitter are both English and Spanish. For this project only the English tweets were used, as the
baseline T5 model was trained on English texts.

The tweets were manually annotated to classify instances of hate speech targeting immigrants and
women. Each instance in the dataset contains the text of the tweet and the corresponding label (0
for no hate speech, 1 for hate speech).

The dataset is structured into training (9,000 instances), validation (1,000 instances), and test
(2,970 instances) sets. The dataset exhibits a relatively balanced distribution, having, on average
across the different sets, a 58% of non hateful tweets and a 42% of hateful tweets.

Examples of annotated instances include:

```
text: your girlfriend lookin at me like a groupie in this bitch!
label: 1

text: Coding program in NY seeks to open tech career doors for immigrant girls
label: 0
```

The training task consists on teaching the T5 model to automatically classify tweets into hate
speech and non-hate speech categories, accounting for the different nuanced forms of discriminatory
language present in the dataset.

Evaluation of model performance utilizes recommended metrics by the original authors, including
macro-accuracy, macro-recall, macro-precision, and macro-F1 Score.

For more information and access to the dataset refer to the Hugging Face's website:
https://huggingface.co/datasets/tweet_eval/
"""

import logging
from typing import cast

from datasets import DatasetDict, load_dataset
from torch import Tensor, device
import torch
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

        self.neg = tokenizer.encode('Hate speech is not present in the previous sentences.')
        self.pos = tokenizer.encode('Hate speech is present in the previous sentences.',
                                    padding="max_length", max_length=len(self.neg))

        dataset_dict = dataset_dict.map(
            lambda x: {"token_labels": self.pos if x["label"] == 1 else self.neg}
        )
        dataset_dict = dataset_dict.rename_column("label", "labels")
        dataset_dict = dataset_dict.map(
            lambda x: tokenizer(x["text"], max_length=157, padding="max_length", truncation=True,
                                return_tensors="pt"), batched=True
        )
        logger.debug("DatasetDict converted successfully.")

        self.train = dataset_dict["train"].with_format("torch", device=device_t)
        self.dev = dataset_dict["validation"].with_format("torch", device=device_t)
        self.test = dataset_dict["test"].with_format("torch", device=device_t)
        logger.info("Dataset tweet_eval loaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    def get_eval_methods(self, device_t: device) -> MetricCollection:
        """
        This method will return an evaluate a combination that contain the corresponding
        metrics to evaluate the dataset.
        """
        metric_collection = MetricCollection([
            MulticlassAccuracy(num_classes=2, average="macro"),
            MulticlassPrecision(num_classes=2, average="macro"),
            MulticlassRecall(num_classes=2, average="macro"),
            MulticlassF1Score(num_classes=2, average="macro")
        ]).to(device_t)
        return metric_collection

    def pre_eval_func(self, batch) -> Tensor:
        """
        This method will return the tensor to be used for evaluation given the outputs
        of the model.

        Negative answers are considered for all cases where it doens't match the
        positive label. However, when computing the loss, these are independent classes.
        """
        output_label = batch.logits.argmax(-1)
        #print(output_label)
        output_label = output_label.tolist()
        #print(self.neg, "\n", output_label[0], "\n")
        output_label = [0 if x == self.neg else 1 for x in output_label]
        # Note for other datasets: After end tokens should not be checked, model doesn't learn it.
        output_label = torch.tensor(output_label).to(batch.logits.get_device())
        return output_label

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
