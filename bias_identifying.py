
import argparse
import sys
import pickle as pkl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
# import wandb
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from craft.craft_torch import Craft, torch_to_numpy
import src.model_utils as mu
import src.concept_utils as cu

# Dataset imports
from src.colour_mnist import get_biased_mnist_dataloader
from src.waterbird import WaterBirdsDataset

# Model imports
from src.resnet18_utils import WaterbirdsResNet18
from src.resnet50_utils import WaterbirdsResNet50

#! /usr/bin/env python3

logging.getLogger('tensorflow').disabled = True
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")
parser.add_argument("--concept_id", type=str, default="0")
parser.add_argument("--layer_depth", type=int, default=4)
parser.add_argument("--number_of_concept", type=int, default=40)
parser.add_argument("--patch_size", type=int, default=8)
parser.add_argument("--concept_dataset_size", type=int, default=3000)
# parser.add_argument("--backprop_step", type=int, default=1000)
# parser.add_argument("--concept_threshold", type=float, default=0.3)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--multi_concept", type=bool, default=False)

args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name
concept_id = args.concept_id
layer_depth = args.layer_depth
number_of_concept = args.number_of_concept
patch_size = args.patch_size
concept_dataset_size = args.concept_dataset_size
# backprop_step = args.backprop_step
backprop_steps = [1, 10, 30, 70, 100, 300, 700, 1000]

# Configuring paths
# data_path = args.data_path
# if data_path == "":
#     data_path = os.getcwd()
# data_path += "/data/" + model_name

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# result_path = args.result_path + "models/"
# result_path += model_name + "/"
# model_result_path = result_path + f"model_{model_id}.pkl"

result_path = os.path.join(args.result_path, "models", model_name)
model_result_path = os.path.join(result_path, f"model_{model_id}.pkl")

if not os.path.exists(model_result_path):
    raise FileExistsError(f"Error ! You need to train the model before being able to test it! Path: {model_result_path}")

with open(model_result_path, "rb") as f:
    model_parameters = pkl.load(f)
batch_size = model_parameters["batch_size"]
split_seed = model_parameters["split_seed"]
shuffle_seed = model_parameters["shuffle_seed"]
dataset = model_parameters["dataset"]
model_type = model_parameters["model_type"]

data_path = args.data_path or os.getcwd()
data_path += "/data/" + dataset

# Specifying dataset loader
if dataset == "MNIST":
    train_loader, val_loader = get_biased_mnist_dataloader(
        root = data_path,
        batch_size = batch_size,
        data_label_correlation = model_parameters["train_correlation"],
        train = True,
        validation = 1/10,
        split_gen_seed = split_seed,
        shuffle_seed = shuffle_seed
    )
if dataset == "Waterbirds":
    train_loader = WaterBirdsDataset(
        root_dir = data_path,
        split = "train"
    )
    val_loader = WaterBirdsDataset(
        root_dir = data_path,
        split = "val"
    )
    train_loader = DataLoader(
        train_loader,
        batch_size = batch_size,
        shuffle = True
    )
    val_loader = DataLoader(
        val_loader,
        batch_size = batch_size,
        shuffle = False
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
if model_type == "MLP":
    model = mu.CMNISTNeuralNetwork()
elif model_type == "resnet18":
    model = WaterbirdsResNet18()
elif model_type == "resnet50":
    model = WaterbirdsResNet50()
else:
    raise ValueError(f"Unknown model type: {model_type}")

model = model.to(device)
model.load_state_dict(torch.load(os.path.join(result_path,f"model_{model_id}")))

loss_fn = nn.CrossEntropyLoss()

# Setting up the concept decomposition
print(f"Using: {device}")
print(f"-Training the concept decomposition", end="\r")

concept_dataset = torch.Tensor([])
for X, *_ in val_loader:
    if len(concept_dataset) >= concept_dataset_size:
        break
    concept_dataset = torch.cat((concept_dataset, X))

# CRAFT layer extraction
if model_name == "MNIST":
    g = nn.Sequential(*(list(model.children())[:layer_depth-1])) # Layers pre concept decomposition
    h = nn.Sequential(*(list(model.children())[layer_depth-1:])) # Layers post concept decompositon
else:
    g = nn.Sequential(*(list(model.backbone.children())[:-1])) # Layers pre concept decomposition
    h = nn.Sequential(model.backbone.fc) # Layers post concept decompositon

craft = Craft(
    input_to_latent=g,
    latent_to_logit = h,
    number_of_concepts = number_of_concept,
    patch_size = patch_size,
    batch_size = batch_size,
    device = device
)

crops, crops_u, w = craft.fit(concept_dataset)
print("-Concept decomposition trained successfully")
crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
gr = cu.Gradient_retriever(h)

results = {}
for backprop_step in backprop_steps:
    print(f"-Gathering concepts with backprop step : {backprop_step}", end="\r")
    results[backprop_step] = cu.gather_all_concept_results(
        val_loader,
        model,
        loss_fn,
        {"all": craft},
        gr,
        backprop_mult = backprop_step,
        device = device
    )
    print(f"-Concepts with backprop step {backprop_step} gathered successfully")

parameters = {
    "model_id":model_id,
    "concept_id":concept_id,
    "layer_depth":layer_depth,
    "number_of_concept":number_of_concept,
    "patch_size":patch_size,
    "concept_dataset_size":concept_dataset_size,
    "backprop_steps":backprop_steps,
    "concept_results": results,
    "concept_parameters": {
        "W": w,
        "crops": crops,
        "crops_u": crops_u,
        "reducer": craft.reducer,
    },
}

out_path = os.path.join(result_path, f"concepts_{model_id}_{concept_id}.pkl")
with open(out_path, "wb") as f:
    pkl.dump(parameters, f)
print(f"OOOOOOOOO Concept experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

