""" This module contains utility functions for working with PyTorch tensors. """

import torch
from peft.peft_model import PeftModel

def get_device() -> torch.device:
    """
    Returns the device to use for PyTorch tensors.

    Returns:
        torch.device: The device to use for PyTorch tensors.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_trainable_params(model: PeftModel) -> list[torch.Tensor]:
    """
    Returns a list of trainable parameters in the current model.

    Returns:
        list[torch.Tensor]: A list of trainable parameters in the current model.
    """
    return [p for p in model.parameters() if p.requires_grad]
