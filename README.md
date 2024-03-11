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

## Example usage

```bash
python main.py -d=tweet_eval -e=2 -ee=300 --debug=DEBUG LoRA -r=2 -a=2 -d=0.1
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