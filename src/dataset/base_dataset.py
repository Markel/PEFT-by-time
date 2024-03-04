"""
This module contains the abstract class for all datasets, so we can do type checking effectively.
"""

import logging
import os
from abc import ABC, abstractmethod

from huggingface_hub import snapshot_download
from datasets.arrow_dataset import Dataset
from datasets.utils.logging import disable_progress_bar
from torch import Tensor, device
from torch.nn import Module
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

        disable_progress_bar()
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
    def pre_eval_func(self, batch) -> Tensor:
        """
        This method will return the tensor to be used for evaluation given the outputs
        of the model.
        """

    @abstractmethod
    def get_loss_function(self) -> Module:
        """
        This method will return the loss function to use for the dataset.
        """

if __name__ == "__main__":
    logger.critical("Module not meant to be run as a script. Exiting.")
