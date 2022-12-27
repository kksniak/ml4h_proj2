# Sentence Classification in Medical Abstracts - Machine Learning for Health Care Project 2

Machine Learning for Health Care Project 1: [Cardiac Abnormality Classification](https://github.com/kksniak/ml4h_project_1.git)

## Table of Contents

- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Content Overview](#content-overview)
- [References](#references)

## Usage

### Environment Setup

1. Create a top-level directory named `data` and put the raw `.txt` data files in it. A detailed diagram is included in the [repository structure](#repository-structure).
1. Download the model and embedding checkpoints from [this link](https://drive.google.com/drive/folders/1EEbfVUyNxlrnMey6P_Mp1LXxBCIW5RDD?usp=sharing). Add them to `code/embeddings_checkpoints` and `code/models/models_checkpoints` respectively.
1. Create a conda environment with `conda env create -f [ENV_NAME].yml`, using the desired environment file. Depending on your environment, the `environment.yml` or `environment-cross-platform.yml` may work better.
1. Activate the conda environment with `conda activate ml4h`

### Running the Code
For reproducing the results from the report, three scripts have been provided, which loads, evaluates, and optionally trains the models. Results are saved in the `results/` directory.

**Note that complete reproducibility is not guaranteed across platforms, as described in the [PyTorch documentation](https://pytorch.org/docs/stable/notes/randomness.html). Thus, you may observe slight deviations in the results compared to those of the report.**

Run the script from the `code` directory with the following commands:

- `python main_TASK1.py`
- `python main_TASK2.py`
- `sh main_TASK3.sh`

To train the models, the `load_model` flags may be switched to `False` in `main_TASK1.py` and `main_TASK2.py`. `main_TASK3.sh` includes code that can be uncommented to evaluate on the full test set or to train the models.

## Repository structure

    .
    ├── code                                
    │   ├── models                      # Classification models
    │   │   ├── models_checkpoints
    │   │   ├── bert.py
    │   │   ├── bidirectional_LSTM_POS.py
    │   │   ├── bidirectional_LSTM.py
    │   │   ├── resnet1d.py
    │   │   ├── vanilla_NN.py
    │   ├── bert_utils.py               # Preprocessing and caching utils for BERT models
    │   ├── config.py                   # Paths to results, checkpoints, etc.
    │   ├── embeddings_analysis.py      # Word similarity analysis
    │   ├── embeddings.py               # Training of embeddings
    │   ├── evaluation.py               # Utils for evaluating model performance
    │   ├── main_TASK1.py               # Main script for task 1 
    │   ├── main_TASK2.py               # Main script for task 2
    │   ├── main_TASK3.sh               # Main script for task 3 
    │   └── utils.py
    ├── data                            # Put `.txt` data here
    ├── results                         # Plots and evaluation metrics
    ├── .gitignore
    ├── .pylintrc                       # Linting config
    ├── .style.yapf                     # Formatter config
    ├── .environment-cross-platform.yml # Cross-platform environment
    └── .environment.yml                # Intel environment

## Content Overview
The following is an overview of the contents of this repository.

- `models/`
    - `models_checkpoints/` – Contains checkpoints for models. Used to avoid retraining.
    - `bert.py` – Module implementing pretrained BERT models + classification layer.
    - `bidirectional_LSTM_POS.py` – Module implementing a bidirectional LSTM including part-of-speech tags.
    - `bidirectional_LSTM.py` – Module implementing a bidirectional LSTM without part-of-speech tags.
    - `resnet1d.py` – Module implementing a ResNet model.
    - `vanilla_NN.py` – Module implementing a vanilla neural network.
- `config.py` – Contains configurable paths for input, output, and checkpoints.
- `embeddings_analysis.py` – Contains code for analyzing word embeddings and word similarities.
- `embeddings.py` – Contains code for training embeddings.
- `evaluation.py` – Utils for evaluation models.
- `main_TASK1.py` – Script for loading and evaluating models for task 1.
- `main_TASK2.py` – Script for loading and evaluating models for task 2.
- `main_TASK3.sh` – Script for loading and evaluating models for task 3.
- `utils.py` – Various utilities.
- `data/` – Directory for raw data.
- `results/` – Directory where plots and metrics are saved by the evaluation utils.

