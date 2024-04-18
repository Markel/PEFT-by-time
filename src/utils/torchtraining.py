"""
This module contains the functions that will train and test the model.
In other words, the training loop that encompasses all the other files.
"""
import logging
import time
from math import ceil
from typing import Callable

import torch
from peft.peft_model import PeftModel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from transformers import T5TokenizerFast

import wandb

from ..dataset.base_dataset import BaseDataset
from .arguments import Args
from .torch_macs import MACCounterMode
from .torchfuncs import (get_results_filename, get_trainable_params, get_optimizer, init_wandb,
                         save_results_file)

logger = logging.getLogger("m.utils.torchtraining")

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
    number_of_shards = 1 if args.eval_every <= 0 else ceil(len(dataset.train)/args.eval_every)
    logger.debug("Dividing the training dataset into %d shards", number_of_shards)

    train_loaders = [
        DataLoader(
            dataset.train.shard(number_of_shards, i, contiguous=True), # type: ignore
            batch_size=args.batch_size, shuffle=False,
        )
        for i in range(number_of_shards)
    ]
    dev_loader    = DataLoader(dataset.dev, # type: ignore
                               batch_size=args.batch_size, shuffle=False)
    test_loader   = DataLoader(dataset.test, # type: ignore
                               batch_size=args.batch_size, shuffle=False)

    logger.debug("Training dataset divided into %d shards", len(train_loaders))
    return ((train_loaders, number_of_shards), dev_loader, test_loader)

def train_entire_batch(model: PeftModel,
                       tokenizer: T5TokenizerFast,
                       data: DataLoader,
                       optimizer: Optimizer,
                       eval_tests: MetricCollection,
                       pre_eval_func: Callable,
                       running_loss: float,
                       ) -> tuple[PeftModel, float, MetricCollection]:
    """
    Trains the model on the entire batch of data passes in the `data` argument.

    Args:
        model (PeftModel): The model to be trained.
        tokenizer (T5TokenizerFast): The tokenizer used for tokenization.
        data (DataLoader): The data loader containing the batch of data.
        optimizer (Optimizer): The optimizer used for gradient updates.
        eval_tests (MetricCollection): The collection of evaluation metrics.
        pre_eval_func (Callable): The function to preprocess the model outputs for evaluation.
        running_loss (float): The running loss value.

    Returns:
        tuple[PeftModel, float, MetricCollection]: A tuple containing the updated model,
        running loss, and evaluation metrics.
    """
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
    """
    Evaluate the model on a batch of test data.

    Args:
        model (PeftModel): The model to be evaluated.
        tokenizer (T5TokenizerFast): The tokenizer used for tokenization.
        data (DataLoader): The data loader containing the test data.
        eval_tests (MetricCollection): The collection of evaluation metrics.
        pre_eval_func (Callable): A function to preprocess the model outputs before evaluation.

    Returns:
        tuple[MetricCollection, float]: A tuple containing the updated evaluation metrics
        and the average loss.
    """
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

def full_training(model: PeftModel,
                  tokenizer: T5TokenizerFast,
                  dataset: BaseDataset,
                  args: Args,
                  device: torch.device
                  ) -> None:
    """
    Performs the full training process for the given model using the provided dataset.

    Args:
        model (PeftModel): The model to be trained.
        tokenizer (T5TokenizerFast): The tokenizer used for tokenizing the input data.
        dataset (BaseDataset): The dataset containing the training, development, and test data.
        args (Args): The program arguments object.
        device (torch.device): The device on which the training will be performed.

    Returns:
        None
    """
    #* INITIALIZATION
    logger.debug("Starting full training")
    run = init_wandb(args)
    logger.info("Weights and Biases run: %s (%s)", run.name, run.id)

    (train_loaders, number_of_shards), dev_loader, test_loader = get_data_loaders(dataset, args)
    run.summary["num_of_shards"] = number_of_shards

    train_params = get_trainable_params(model)
    num_trainable_params = sum(p.numel() for p in train_params)
    logger.debug("Number of trainable parameters: %d", num_trainable_params)
    run.summary["num_of_trainable_params"] = num_trainable_params
    run.summary["total_params"] = sum(p.numel() for p in model.parameters())

    optimizer = get_optimizer(model, args)

    train_tests = dataset.get_eval_methods(device)
    dev_tests   = dataset.get_eval_methods(device)
    test_tests  = dataset.get_eval_methods(device)

    iters_need = number_of_shards * args.epochs
    logger.debug("%d iterations are going to be required", iters_need)

    steps_done : int = 0
    time_done  : float = 0.0 # In seconds
    gmacs_done: float = 0.0

    running_loss: float = 0.0

    if not args.skip_initial_eval:
        logger.info("Starting initial evaluation")
        logger.debug("Starting to evaluate dev")
        dev_tests, dev_loss = test_batch(model, tokenizer, dev_loader,
                                         dev_tests, dataset.pre_eval_func)
        dev_results = dev_tests.compute()
        dev_tests.reset()
        dev_results = {f"dev_{key}": value.item() for key, value in dev_results.items()}

        logger.debug("Starting to evaluate test")
        test_tests, test_loss = test_batch(model, tokenizer, test_loader,
                                           test_tests, dataset.pre_eval_func)
        test_results = test_tests.compute()
        test_tests.reset()
        test_results = {f"test_{key}": value.item() for key, value in test_results.items()}

        results = {**dev_results, **test_results, "dev_loss": dev_loss, "test_loss": test_loss,
                   "steps_done": 0, "iteration": -1, "epochs_done": 0, "time_in_train": 0.0,
                   "GMACs": 0.0}
        wandb.log(results)
        save_results_file(results, run.name, run.id)
        logger.info("Initial evaluation done. Results: %s", results)

    #* WORKING LOOP
    for iteration in range(iters_need):
        loader_index = iteration % number_of_shards
        logger.debug("Iteration %d, loader index %d", iteration, loader_index)

        #* If iteration 0 reset train tester as an epoch has been completed
        if loader_index == 0:
            train_tests.reset()
            running_loss = 0.0

        #* TRAINING
        logger.debug("Starting to train, iteration %d", iteration)

        m_counter = MACCounterMode(model, show = iteration==0)
        with m_counter:
            start_time = time.time()
            model, running_loss, train_tests = train_entire_batch(model, tokenizer,
                                                                  train_loaders[loader_index],
                                                                  optimizer, train_tests,
                                                                  dataset.pre_eval_func,
                                                                  running_loss)
            end_time = time.time()
        gmacs_done += m_counter.get_total()
        if iteration == 0:
            run.summary["macs_per_step"] = (m_counter.get_total(divided=False)
                                            / len(train_loaders[0].dataset)) # type: ignore
        steps_done += len(train_loaders[loader_index].dataset) # type: ignore
        time_done += (end_time - start_time)

        train_results = train_tests.compute()
        train_results = {f"train_{key}": value.item() for key, value in train_results.items()}

        #* EVALUATION
        logger.debug("Starting to evaluate dev, iteration %d", iteration)
        dev_tests, dev_loss = test_batch(model, tokenizer, dev_loader,
                                         dev_tests, dataset.pre_eval_func)
        dev_results = dev_tests.compute()
        dev_tests.reset()
        dev_results = {f"dev_{key}": value.item() for key, value in dev_results.items()}

        logger.debug("Starting to evaluate test, iteration %d", iteration)
        test_tests, test_loss = test_batch(model, tokenizer, test_loader,
                                           test_tests, dataset.pre_eval_func)
        test_results = test_tests.compute()
        test_tests.reset()
        test_results = {f"test_{key}": value.item() for key, value in test_results.items()}

        #* LOGGING
        results = {**train_results, **dev_results, **test_results,
                   "train_loss": running_loss/steps_done,
                   "dev_loss": dev_loss,
                   "test_loss": test_loss,
                   # "learning_rate": optimizer.param_groups[0]["lr"],
                   "steps_done": steps_done, "iteration": iteration,
                   "epochs_done": (iteration + 1) // number_of_shards,
                   "time_in_train": time_done, "GMACs": gmacs_done}

        save_results_file(results, run.name, run.id)
        wandb.log(results)

        if loader_index == number_of_shards - 1:
            logger.info("Epoch %d done. Results: %s", iteration // number_of_shards, results)

    logger.info("Training finished, procedding to end the run.")
    logger.debug("Logging results file to W&B")
    artifact = wandb.Artifact(name="results_file", type="results")
    artifact.add_file(local_path=get_results_filename(run.name, run.id))
    run.log_artifact(artifact)
    logger.debug("Results file logged to W&B. Ending run.")
    run.finish()
