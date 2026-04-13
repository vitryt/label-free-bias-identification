# Label-Free Bias Identification

A framework for automatically identifying and mitigating biases in vision classification models using [CRAFT](https://github.com/deel-ai/xplique)-based concept explainability methods — without requiring ground-truth bias labels.

## Overview

This repository implements a pipeline that:

1. **Trains** a biased image classification model on a spuriously correlated dataset.
2. **Identifies** concepts learned by the model using Non-negative Matrix Factorization (NMF) decomposition of intermediate activations (via CRAFT).
3. **Analyses** which concepts are associated with model errors (false positives / false negatives) using gradient-based importance scoring.
4. **Debiases** the model at inference time by suppressing bias-related concept activations.

## Supported Datasets

| Name | Key | Description |
|------|-----|-------------|
| Coloured MNIST | `MNIST` | Digit classification with spurious colour-digit correlations |
| Waterbirds | `Waterbirds` | Bird classification with spurious background correlations |
| CelebA | `CelebA` | Celebrity attribute classification |
<!-- | UrbanCars | `UrbanCars` | Urban car classification | -->

Datasets were installed from the following links :
- CMNIST: automatic
- [Waterbirds](https://github.com/kohpangwei/group_DRO)
- [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Supported Models

| Key | Architecture |
|-----|-------------|
| `MLP` | Multi-layer perceptron (for CMNIST) |
| `cnn` | Small CNN (for CMNIST) |
| `resnet18` | ResNet-18 (for Waterbirds) |
| `resnet50` | ResNet-50 (for Waterbirds) |
| `resnetceleb` | ResNet-50 (for CelebA) |

## Repository Structure

```
.
├── src/
│   ├── colour_mnist.py      # Coloured MNIST dataset & model definition
│   ├── waterbird.py         # Waterbirds dataset & ResNet model definitions
│   ├── celeba.py            # CelebA dataset & model definition
│   ├── dataset_utils.py     # Unified dataloader factory
│   ├── model_utils.py       # Training/evaluation loops, model & optimizer factory
│   ├── concept_utils.py     # CRAFT-based concept decomposition & bias analysis utilities
│   └── nmf_merge.py         # NMF component-level fusion across decompositions
├── bash_scripts/            # Folder with various pipeline scripts to train and evaluate 10 models on various datasets
├── figures/                 # Folder with the figures from the paper. All .png can be recreated with `results_recovery.ipynb`
├── environment.yml          # Environment file for Conda
├── requirements.txt         # Environment file for Pip
├── model_training.py        # Step 1 – Train a classification model
├── bias_identifying.py      # Step 2 – Decompose activations into concepts
├── bias_analysis.py         # Step 2.5 – Score concepts for bias association (for CMNIST only)
├── debiasing.py             # Step 3 – Suppress bias concepts at inference
├── results_recovery.ipynb   # Notebook for dataset preparation / inspection
└── novic.ipynb              # Notebook to us with NOVIC to recover the generated labels of the concepts
```

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate label-free-bias
pip install Craft-xai
```

### pip

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `Craft-xai`, `scikit-learn`, `numpy`, `wandb`.

## Pipeline Usage

Each step saves its results as a `.pkl` file and skips re-execution if the output already exists.

### 1. Train a model

```bash
python model_training.py \
    --model_id 0 \
    --model_name CMNISTb \
    --dataset MNIST \
    --model_type MLP \
    --optimizer adam \
    --batch_size 256 \
    --epochs 20 \
    --train_correlation 0.95 \
    --test_correlation 0.1 \
    --split_seed 42 \
    --shuffle_seed 42 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### 2. Identify concepts (bias candidates)

```bash
python bias_identifying.py \
    --model_id 0 \
    --model_name CMNISTb \
    --concept_id b_0_0 \
    --layer_depth 4 \
    --number_of_concept 16 \
    --patch_size 8 \
    --concept_dataset_size 1000 \
    --multi_concept 1 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### 2.5. Analyse biases

```bash
python bias_analysis.py \
    --model_id 0 \
    --model_name CMNISTb \
    --concept_id b_0_0 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### 3. Debias at inference

```bash
python debiasing.py \
    --model_id 0 \
    --model_name CMNIST \
    --concept_id b_0_0 \
    --bias_threshold 55 \
    --backprop_step 20000 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### Running the full pipeline with provided scripts

```bash
# CMNIST / MLP
bash bash_scripts/scriptb.sh
bash bash_scripts/scriptbb.sh

# Waterbirds / ResNet-18
bash bash_scripts/scriptwater18.sh

# CelebA / ResNet-50
bash bash_scripts/scriptceleb.sh
```

## Output files

All outputs are written under `<result_path>/models/<model_name>/`:

| File | Content |
|------|---------|
| `model_<id>.pkl` | Training hyperparameters and evaluation matrices |
| `model_<id>` | Saved model weights (`torch.save`) |
| `concepts_<model_id>_<concept_id>.pkl` | Concept decomposition and importance scores |
| `debias_<model_id>_<concept_id>.pkl` | Merged concept dictionary and debiased evaluation results |

## NOVIC

In order to run the novic.ipynb code, we advise to get and follow the instructions from the [NOVIC repository](https://github.com/pallgeuer/novic) in a new environment. The pip version was found more stable.