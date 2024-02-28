import logging
from math import ceil
from typing import cast
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from peft.peft_model import PeftModel
from transformers import T5TokenizerFast
from wandb.sdk.wandb_run import Run

import wandb

from ..utils.torchfuncs import get_trainable_params

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

def train_entire(model: PeftModel, tokenizer: T5TokenizerFast, data: DataLoader, optimizer: Optimizer, loss, args: Args) -> None:
    model.train()



def full_training(model: PeftModel, tokenizer: T5TokenizerFast, dataset: BaseDataset, args: Args) -> None:
    logger.debug("Starting full training")
    run = init_wandb(args)
    logger.info("Weights and Biases run: %s (%s)", run.name, run.id)

    number_of_shards = ceil(len(dataset.train)/args.eval_every)
    logger.debug("Dividing the training dataset into %d shards", number_of_shards)

    train_loaders = [
        DataLoader(
            dataset.train.shard(number_of_shards, i, contiguous=True), # type: ignore
            batch_size=args.batch_size, shuffle=False
        )
        for i in range(number_of_shards)
    ]
    dev_loader   = DataLoader(dataset.dev, # type: ignore
                              batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(dataset.test, # type: ignore
                              batch_size=args.batch_size, shuffle=False)

    loss = dataset.get_loss_function() # There's no specific type :(
    optimizer = torch.optim.Adam(get_trainable_params(model), lr=args.learning_rate)

    datasetpost = dataset.train.map(lambda examples: {"labels": examples["label"]}, batched=True)

    iters_need = number_of_shards * args.epochs
    logger.debug("%d iterations are going to be required", iters_need)
    #steps_eval = args.eval_every if args.eval_every > 0 else len(dataset.train)

    steps_done: int = 0
    time_done: int = 0

    for iter in range(iters_need):
        loader_index = iter % number_of_shards
        for data in train_loaders[loader_index]:
            print(data)
        break
        #print(f"Step {step_init} - {cut_final}/{iters_need}")


    wandb.log({"train_loss": 0.2, "dev_loss": 0.3, "test_loss": 0.1})

    run.finish()
    return
"""
    wandb.watch(model, log="all")
    model.train()
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            wandb.log({"loss": loss.item()})
    wandb.finish()
"""
"""
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
"""