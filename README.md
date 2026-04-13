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
| UrbanCars | `UrbanCars` | Urban car classification |

## Supported Models

| Key | Architecture |
|-----|-------------|
| `MLP` | Multi-layer perceptron (for CMNIST) |
| `resnet18` | ResNet-18 (for Waterbirds) |
| `resnet50` | ResNet-50 (for Waterbirds) |
| `resnetceleb` | ResNet-50 (for CelebA) |
| `resneturban18` | ResNet-18 (for UrbanCars) |
| `resneturban50` | ResNet-50 (for UrbanCars) |

## Repository Structure

```
.
├── src/
│   ├── colour_mnist.py      # Coloured MNIST dataset & model definition
│   ├── waterbird.py         # Waterbirds dataset & ResNet model definitions
│   ├── celeba.py            # CelebA dataset & model definition
│   ├── urbancars.py         # UrbanCars dataset & model definitions
│   ├── dataset_utils.py     # Unified dataloader factory
│   ├── model_utils.py       # Training/evaluation loops, model & optimizer factory
│   ├── concept_utils.py     # CRAFT-based concept decomposition & bias analysis utilities
│   └── nmf_merge.py         # NMF component-level fusion across decompositions
├── model_training.py        # Step 1 – Train a classification model
├── model_testing.py         # Step 2 – Evaluate model on biased test sets
├── bias_identifying.py      # Step 3 – Decompose activations into concepts
├── bias_analysis.py         # Step 4 – Score concepts for bias association
├── debiasing.py             # Step 5 – Suppress bias concepts at inference
├── data_recovery.ipynb      # Notebook for dataset preparation / inspection
├── script.sh                # Full pipeline script for CMNIST / MLP
├── script18.sh              # Full pipeline script for Waterbirds / ResNet-18
├── script50.sh              # Full pipeline script for Waterbirds / ResNet-50
├── environment.yml          # Conda environment specification
└── requirements.txt         # pip requirements
```

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate debiafting
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
    --model_name CMNIST \
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

### 2. Test the model

```bash
python model_testing.py \
    --model_id 0 \
    --model_name CMNIST \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### 3. Identify concepts (bias candidates)

```bash
python bias_identifying.py \
    --model_id 0 \
    --model_name CMNIST \
    --concept_id 0_0 \
    --layer_depth 4 \
    --number_of_concept 16 \
    --patch_size 8 \
    --concept_dataset_size 1000 \
    --multi_concept 1 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### 4. Analyse biases

```bash
python bias_analysis.py \
    --model_id 0 \
    --model_name CMNIST \
    --concept_id 0_0 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### 5. Debias at inference

```bash
python debiasing.py \
    --model_id 0 \
    --model_name CMNIST \
    --concept_id 0_0 \
    --bias_threshold 30 \
    --backprop_step 700 \
    --result_path /path/to/results/ \
    --data_path /path/to/data/
```

### Running the full pipeline with provided scripts

```bash
# CMNIST / MLP
bash script.sh

# Waterbirds / ResNet-18
bash script18.sh

# Waterbirds / ResNet-50
bash script50.sh
```

## Output files

All outputs are written under `<result_path>/models/<model_name>/`:

| File | Content |
|------|---------|
| `model_<id>.pkl` | Training hyperparameters and evaluation matrices |
| `model_<id>` | Saved model weights (`torch.save`) |
| `test_model_<id>.pkl` | Adjacency matrix from biased test evaluation |
| `concepts_<model_id>_<concept_id>.pkl` | Concept decomposition and importance scores |
| `bias_<model_id>_<concept_id>.pkl` | Per-concept bias scores per class |
| `debias_<model_id>_<concept_id>.pkl` | Merged concept dictionary and debiased evaluation results |
