# PEFT Efficiency through time (temporary title)

## Instalation

Tested Python version: 3.9

Tested CUDA version: 12.3

It's recommended to first create a virtual environment for the project:
```bash
python -m venv peft-env
```

Install the dependencies:
```bash
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

Create a .env file and set your Weights & Biases key. An fictional example is provided here:
```
WANDB_API_KEY=5B5FCFA35882C50972C4CF6AA89DBFCA19608A28
```

## Usage

```
python main.py [-h] [-b BATCH_SIZE] -d {tweet_eval} [--debug {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [-e EPOCHS] [-ee EVAL_EVERY] [-lr LEARNING_RATE] [-m MODEL] [-nc] [-Wen EXPERIMENT_NAME] [-Wp PROJECT] [-Wt RUN_TYPE] {LoRA,FT}
```

| Argument flag           | Description                                                                                                                                                     | Default           |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| -h                      | Show help                                                                                                                                                       | False             |
| -b, --batch_size        | Modifies the batch size used in training and testing                                                                                                            | 8                 |
| -d, --dataset           | Selects the desired dataset to test on.                                                                                                                         | Required argument |
| --debug                 | The log level to set the logger to.                                                                                                                             | INFO              |
| -e, --epochs            | How many epochs will the model be trained on.                                                                                                                   | 10                |
| -ee, --eval_every       | Every how many steps will the training stop and do a round of testing.                                                                                          | 1 epoch.          |
| -lr, --learning_rate    | Learning rate of the optimizer.                                                                                                                                 | 1e-3              |
| -m, --model             | The model to use. "small", "base", "large", "xl", "xxl" default to the corresponding T5v1.1 versions. Other models should be passed in huggingface repo format. | base              |
| -nc, --no_color         | Disables the color from the logger.                                                                                                                             | False             |
| -o, --optimizer         | Chooses the optimizer to use in training. Possible options are "adafactor" and "adam"                                                                           | adafactor         | 
| -Wen, --experiment_name | Sets a custom run name for Weights&Biases                                                                                                                       |                   |
| -Wp, --project          | The Weights&Biases project name.                                                                                                                                | peft-by-time      |
| -Wt, --run_type         | Custom parameter for later filtering in Weights&Biases                                                                                                          | run_test          |
| PEFT_method             | Select the PEFT method to use.                                                                                                                                  | Required argument |

### LoRA

Adapt the model to use [Low Rank Adaptation](https://arxiv.org/abs/2106.09685).

```
python main.py ... LoRA [-h] [-r RANK] [-a ALPHA] [-d DROPOUT] [-t TARGET_MODULES [TARGET_MODULES ...]]
```

| Argument flag        | Description                                                       | Default |
|----------------------|-------------------------------------------------------------------|---------|
| -h                   | Show help                                                         | False   |
| -r, --rank           | The rank of the LoRA matrices.                                    | 2       |
| -a, --alpha          | The alpha parameter for LoRA                                      | 2       |
| -d, --dropout        | How much dropout to apply.                                        | 0.1     |
| -t, --target_modules | Which matrices should be targeted by LoRA. Multiple values field. | [q, v]  |

### Usage examples

```bash
python main.py -d=tweet_eval -e=2 -ee=300 --debug=DEBUG LoRA
```

## Linting
```bash
pylint ./
```

## Disable reporting to W&B

You may disable Weigths & Bias reporting (it will be treated as a dummy run) creating a .env file and setting

```
WANDB_MODE=disabled
```