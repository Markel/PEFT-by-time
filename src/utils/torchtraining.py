import logging
from math import ceil
from typing import Callable, cast
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from peft.peft_model import PeftModel
from torchmetrics import MetricCollection
from transformers import T5TokenizerFast
from wandb.sdk.wandb_run import Run

import wandb

from ..utils.torchfuncs import get_trainable_params, save_results_file
from ..utils.torch_flops import FlopCounterMode

from ..dataset.base_dataset import BaseDataset
from .arguments import Args

logger = logging.getLogger("m.utils.torchtraining")

def get_wandb_experiment_name(args: Args) -> str:
    """
    Returns the experiment name for logging to Weights & Biases (wandb).

    Args:
        args (Args): The command line arguments.

    Returns:
        str: The experiment name.
    """
    if args.experiment_name is not None:
        return args.experiment_name
    return f"{args.method}_{args.model}_{args.dataset}"

def init_wandb(args: Args) -> Run:
    """
    Initializes and returns a Weights & Biases run object.

    Args:
        args (Args): An instance of the Args class containing the configuration parameters.

    Returns:
        Run: A Weights & Biases run object.

    """
    args_dict = args.__dict__.copy()
    del args_dict["project"], args_dict["experiment_name"]
    my_run = wandb.init(
        project=args.project,
        name=get_wandb_experiment_name(args),
        config=args_dict
    )
    return cast(Run, my_run)

def train_entire_batch(model: PeftModel,
                       tokenizer: T5TokenizerFast,
                       data: DataLoader,
                       optimizer: Optimizer,
                       eval_tests: MetricCollection,
                       pre_eval_func: Callable,
                       running_loss: float,
                       ) -> tuple[PeftModel, float, MetricCollection]:
    model.train()
    for batch in data:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_labels   = batch["token_labels"]
        labels         = batch["labels"]
        token_labels[token_labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=token_labels)
        loss = outputs.loss
        loss.backward()
        running_loss += loss.item()
        output_labels = pre_eval_func(outputs)
        #print("Ou", output_labels, "LA", labels)
        eval_tests.update(output_labels, labels)
        optimizer.step()
    return model, running_loss, eval_tests

def test_batch(model: PeftModel,
               tokenizer: T5TokenizerFast,
               data: DataLoader,
               eval_tests: MetricCollection,
               pre_eval_func: Callable
               ) -> tuple[MetricCollection, float]:
    model.eval()
    total_loss = 0.0
    for batch in data:
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_labels   = batch["token_labels"]
        labels         = batch["labels"]
        token_labels[token_labels == tokenizer.pad_token_id] = -100

        # Mirar de cambiar a inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=token_labels)
            loss = outputs.loss
            total_loss += loss.item()
        output_labels = pre_eval_func(outputs)
        eval_tests.update(output_labels, labels)
    total_loss /= len(data.dataset) # type: ignore
    return eval_tests, total_loss

def get_data_loaders(dataset: BaseDataset,
                     args: Args
                     ) -> tuple[tuple[list[DataLoader], int], DataLoader, DataLoader]:
    """
    Get data loaders for training, development, and testing. The train loader will be split
    into shards for easier training by steps.

    Args:
        dataset (BaseDataset): The dataset object containing train, dev, and test data.
        args (Args): The arguments object containing batch size and evaluation frequency.

    Returns:
        tuple: A tuple containing:
            - A tuple of train loaders and the number of shards.
            - The development loader.
            - The test loader.
    """
    number_of_shards = ceil(len(dataset.train)/args.eval_every)
    logger.debug("Dividing the training dataset into %d shards", number_of_shards)

    train_loaders = [
        DataLoader(
            dataset.train.shard(number_of_shards, i, contiguous=True), # type: ignore
            batch_size=args.batch_size, shuffle=False
        )
        for i in range(number_of_shards)
    ]
    dev_loader    = DataLoader(dataset.dev, # type: ignore
                               batch_size=args.batch_size, shuffle=False)
    test_loader   = DataLoader(dataset.test, # type: ignore
                               batch_size=args.batch_size, shuffle=False)

    return ((train_loaders, number_of_shards), dev_loader, test_loader)

def full_training(model: PeftModel,
                  tokenizer: T5TokenizerFast,
                  dataset: BaseDataset,
                  args: Args,
                  device: torch.device
                  ) -> None:
    logger.debug("Starting full training")
    run = init_wandb(args)
    logger.info("Weights and Biases run: %s (%s)", run.name, run.id)

    (train_loaders, number_of_shards), dev_loader, test_loader = get_data_loaders(dataset, args)
    loss = dataset.get_loss_function() # There's no specific type :(
    # TODO: Remove loss as CrossEntropy is used internally
    
    train_params = get_trainable_params(model)
    num_trainable_params = sum(p.numel() for p in train_params)
    logger.debug("Number of trainable parameters: %d", num_trainable_params)
    optimizer = torch.optim.Adam(get_trainable_params(model), lr=args.learning_rate)
    # TODO: Maybe AdaFactor? As the original paper.

    train_tests = dataset.get_eval_methods(device)
    dev_tests   = dataset.get_eval_methods(device)
    test_tests  = dataset.get_eval_methods(device)

    iters_need = number_of_shards * args.epochs
    logger.debug("%d iterations are going to be required", iters_need)

    steps_done : int = 0
    time_done  : float = 0.0 # In seconds
    GFlops_done: float = 0.0

    running_loss: float = 0.0

    for iteration in range(iters_need):
        loader_index = iteration % number_of_shards
        logger.debug("Iteration %d, loader index %d", iteration, loader_index)
        #* If iteration 0 reset train tester
        if loader_index == 0:
            train_tests.reset()
            running_loss = 0.0

        logger.debug("Starting to train, iteration %d", iteration)
        
        f_counter = FlopCounterMode(model)
        with f_counter:
            start_time = time.time()
            model, running_loss, train_tests = train_entire_batch(model, tokenizer, train_loaders[loader_index], optimizer, train_tests, dataset.pre_eval_func, running_loss)
            end_time = time.time()
        GFlops_done += f_counter.get_total()
        steps_done += len(train_loaders[loader_index].dataset) # type: ignore
        time_done += (end_time - start_time)

        train_results = train_tests.compute()
        train_results = {f"train_{key}": value.item() for key, value in train_results.items()}

        logger.debug("Starting to evaluate dev, iteration %d", iteration)
        dev_tests, dev_loss = test_batch(model, tokenizer, dev_loader, dev_tests, dataset.pre_eval_func)
        dev_results = dev_tests.compute()
        dev_tests.reset()
        dev_results = {f"dev_{key}": value.item() for key, value in dev_results.items()}

        logger.debug("Starting to evaluate test, iteration %d", iteration)
        test_tests, test_loss = test_batch(model, tokenizer, test_loader, test_tests, dataset.pre_eval_func)
        test_results = test_tests.compute()
        test_tests.reset()
        test_results = {f"test_{key}": value.item() for key, value in test_results.items()}

        results = {**train_results, **dev_results, **test_results,
                   "train_loss": running_loss/steps_done,
                   "dev_loss": dev_loss,
                   "test_loss": test_loss,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "step": steps_done, "iteration": iteration,
                   "epoch": iteration // number_of_shards,
                   "time": time_done, "GFlops": GFlops_done}
        # TODO: Cambiar nombres a steps y así

        save_results_file(results, run.name, run.id)
        wandb.log(results) # TODO: Check if convert to int is needed for x axis

        if loader_index == number_of_shards - 1:
            logger.info("Epoch %d done. Results: %s", iteration // number_of_shards, results)
        # TODO: Check -ee=300 (what's happening)
        # TODO: Check FLOP calculation
        # TODO: Document the code
        # TODO: Listen to pylint
        # TODO: Some kind of model saving?
        # TODO: Adam LR - scheduler
    run.finish()
    return
