<h1 style="text-align:center" align="center">Comparison of learning-efficiency using PEFT techniques</h1>

<p style="text-align:center" align="center">
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python"/></a>
<a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=flat&logo=pytorch" alt="PyTorch"/></a>
<a href="https://addi.ehu.es/handle/10810/"><img  alt="ADDI" src="https://img.shields.io/badge/ADDI-waiting to publish-burlywood.svg?logo=data:image/png%2bxml;base64,iVBORw0KGgoAAAANSUhEUgAAAIgAAABiCAQAAAAhbFM9AAABJGlDQ1BJQ0MgcHJvZmlsZQAAKJGdkD1Lw1AUhp9WqUV08mMoDhlcO9pBHPzC0KFQ2wpGp/QmxWISQ5JS/Af+E/0xHQTB3+Cs4Ox7o4ODWbxweB8O57zvvRfqTmTifPkQ4qTI3MGRd+ldOStvNNiiyT4t3+Rpb3g2ovJ8vlKz+tK2XtVzf55GEOZGulAlJs0KqB2IO/Mitaxi83Y0OBE/iJ0gTgLxk3g3iAPLdncQRzPz42lvsxYmF0PbV+3g0qVHH4cxM6ZEFLSliTqndNiTumT43JNjpBGhenPNFNyIcjm5HItGIt2mIq9V5vWVMpbHVF424Y5YnjYP+7/fax/n5WZte5H6mV+2llT1yQTeH2Hdg41nWL2uyGr+flvFTKec+ecbvwBXzlCcszgubgAAAAJiS0dEAACqjSMyAAAACXBIWXMAAC4jAAAuIwF4pT92AAAAB3RJTUUH6AYVCQQZ0ZI3CQAAArdJREFUeNrtm8tRAzEMhqVlr0xogDMMaSAN0EBSAQWEAqABGqAPuDIDFaQA0gAVcAVxyIMlZB9eS1rZKx2TGUf+LP2SHwFwc+toNAegcLvLiwICABBFDYE5ASmSjOVzAACaUKP39DiSCAnzNtS7AjK30MXOHkgokhEACUMyCiAhSEYCBICIrpWAbBu0uXkmL9V+UrDsapdlCW9/fVQAwo1Fxtudf2pArEPZ+bbRkFubTVKT+1Vj39ylGCPcwHeelVlt3RkEYDR9iEkgKZycqAKhmX3sbKKK2D6GVoT0mQ2zqCLySFomKVNdee6+QNtKmSyWiZNjo3LjL3jjo9FWSfQyfD2e/Fp2HTVGVJNvzKgwpiHH1x319khfgJwaJryXSa/iFDlOOOY4MzK0dYEc9/K/D4OJal7X3CoaYnXnIqIhOcaHHxA5EEEgqW/0BSJE/YXZVFGpI+xhP8pT+93pgH42GuNuV7MGyaXpgPcy1UnZK91iVYYmNZ/PWlNgOSQQ9ZTp/kvdoofLc/EDolSLsuHGbPM8N5uUqQv50aZMqknne5mDaCxy3spXbFE/w793jaU0cTurv78JeIeL7Vcn+F0jqnw5G4Ki7Re7jMUxRo2GICLCWe/L6ufwB3CIiHBvTE24lL9/kvQZsxL2zMlbWsjxOiREsMZL3fJrIEJkOw7/R5U3Zg4kDSD0arslVxdVmcKrDyQ0Qm6aprW3j/1nV1rPdbmWCHVXM/aMRH4/5aJqr1O1tdNOLUJW2Gq5R8gnnpruQ5SPft50cfSoMnFVIbTK6J+8FT2ntbYvj4oR0j9OwiJkCCBRj+5yPKuPfsl8UPKmqQMZ4FnLYVxZktTBH8xklzKuIQ7EgXDYwquMiHAm1ph5yjgQB+LmQByIA3EgDsSBOBAH4kAciMm9rqjRcpi/ALm5uSnaDxOFuuFd8UTHAAAAAElFTkSuQmCC"/></a>
</p>

<p style="text-align:center" align="center">
This project helps to compare the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) techniques in various NLP tasks. PEFT methods can democratize access to high-performance LLM models by making their fine-tuning more resource-efficient and accessible for diverse applications. The project implements three PEFT methods: Prefix-Tuning, Parallel Adapters, and Low-Rank Adaptation (LoRA). It can experiment on tasks including emotion classification, topic classification, multiple-choice question answering, and distractor generation, the research aims to identify the most effective PEFT technique given any number of operations.
</p>

## Requirements

Tested Python version: 3.9

Tested CUDA version: 12.3


## Installation
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

The `main.py` file is the starting point of the project, obtain help with:

```
python main.py -h
```

Some example of usage are the following:

```bash
python main.py -d=commonsense_qa -e=4 -ee=100 FT
```

It executes a normal fine-tuning experiment on the CommonsenseQA dataset for 4 epochs with evaluation every 100 steps.

```bash
python main.py -d=race -e=1 -ee=100 LoRA -r=8 -a=8
```

It executes a LoRA experiment on the RACE dataset with a rank of 8 and alpha of 8. It runs for 1 epoch with evaluation every 100 steps.

```bash
python main.py -d=ag_news -e=1 -ee=100 prefix -nt=100
```

It executes a Prefix-Tuning experiment on the AG News dataset with a prefix length of 100. It runs for 1 epoch with evaluation every 100 steps.

```bash
python main.py -m=large -d=commonsense_qa --debug=DEBUG -e=4 -ee=100 adapters -rf=8
```

It executes a Parallel Adapters experiment on the CommonsenseQA dataset with a reduction factor of 8. It runs for 4 epochs with evaluation every 100 steps. It executes the large version of the T5 model and sets the logging level to DEBUG.

### Linting
```bash
pylint ./
```

### Disable reporting to W&B

You may disable Weigths & Bias reporting (it will be treated as a dummy run) creating a .env file and setting

```
WANDB_MODE=disabled
```

## Architecture

The project is structured as follows:

- `assets`: Contains images and other assets used for explaining the project.
- `downloads`: It is created when the program is executed and contains the downloaded models and datasets.
- `plotting`: Contains scripts to plot the results of the experiments.
- `results`: The results of the experiments are stored here. Our results are stored in the repository for reference.
- `src`: Contains the source code of the project.

The source code is structured as follows:

- `dataset`: Contains the dataset base class and the implementations of the datasets used in the experiments.
- `models`: Contains the code for downloading the model and applying the PEFT techniques.
- `utils`: Contains the training functions and other auxiliary functions.

A diagram of the program flow is shown below:

![Diagram of the program flow](https://github.com/Markel/PEFT-by-time/blob/main/assets/PEFT.drawio.png?raw=true)
