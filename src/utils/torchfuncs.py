""" This module contains utility functions for working with PyTorch tensors. """

import json
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

def load_results_file(run_name: str, run_id: str) -> list:
    """
    Load the contents of a results file in JSON format.

    Args:
        filename (str): The path to the results file.

    Returns:
        list: The contents of the results file as a list,
        or an empty list if the file is not found.
    """
    try:
        filename = f"results/{run_name}-{run_id}.json"
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_results_file(dictionary_list: dict, run_name: str, run_id: str):
    """
    Will add the new results to the existing results and resave them to a file.

    Args:
        dictionary_list (dict): The new results to save.
        run_name (str): The name of the run.
        run_id (str): The ID of the run.

    Returns:
        None
    """
    existing_dicts = load_results_file(run_name, run_id)
    existing_dicts.append(dictionary_list)
    filename = f"results/{run_name}-{run_id}.json"
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(existing_dicts, file, indent=2)
