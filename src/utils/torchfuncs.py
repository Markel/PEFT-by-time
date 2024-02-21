""" This module contains utility functions for working with PyTorch tensors. """

import torch

def get_device() -> torch.device:
    """
    Returns the device to use for PyTorch tensors.

    Returns:
        torch.device: The device to use for PyTorch tensors.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
