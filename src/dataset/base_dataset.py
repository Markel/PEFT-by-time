"""
This module contains the abstract class for all datasets, so we can do type checking effectively.

The dataset should have a train, dev and test attribute, with each split.
Each dataset should have the following properties:
input_ids      = The input ids of the text inputs.
attention_mask = The attention mask of the text inputs.
labels         = The number of the results, for example: 0 non-hate, 1 hate.
token_labels   = A phrase that represents the label. This is what the model will predict.
                 For example: "The topic of the article is: World"
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Union

from huggingface_hub import snapshot_download
from datasets.arrow_dataset import Dataset
from datasets.utils.logging import disable_progress_bar
from torch import Tensor, device
from torchmetrics import MetricCollection

logger = logging.getLogger("m.dataset.base_dataset")

class BaseDataset(ABC):
    """
    This is an abstract class for having typing available for the different dataset.
    SHOULD NOT be instanciated directly, higher hierarchy classes should be used.
    """
    train: Dataset
    dev: Dataset
    test: Dataset

    @abstractmethod
    def __init__(self, dataset_name: str) -> None:
        super().__init__()

        disable_progress_bar() # Doesn't really work...
        # Download the dataset if not available
        local_dir = "./downloads/datasets/"+dataset_name
        if not os.path.isdir(local_dir):
            logger.info("Downloading the dataset from the internet.")
            snapshot_download(dataset_name, local_dir=local_dir, repo_type="dataset")
            logger.debug("Dataset downloaded successfully.")

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_eval_methods(self, device_t: device) -> MetricCollection:
        """
        This method will return an evaluate a combination that contain the corresponding
        metrics to evaluate the dataset. Note that this combination may be of len(1)
        """

    @abstractmethod
    def pre_eval_func(self, batch, labels) -> Union[Tensor, list[str]]:
        """
        This method will return the tensor to be used for evaluation given the outputs
        of the model.
        """

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
