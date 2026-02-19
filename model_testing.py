
import argparse
import sys
import pickle as pkl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random

import torch
from src.colour_mnist import get_biased_mnist_dataloader
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import src.model_utils as mu
import src.concept_utils as cu

# import wandb

import numpy as np

import logging
logging.getLogger('tensorflow').disabled = True

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")

parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")


args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name


data_path = args.data_path
if data_path == "":
    data_path = os.getcwd()
data_path += "/data/" + model_name

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

result_path = args.result_path + "models/"
result_path += model_name + "/"

model_result_path = result_path + f"model_{model_id}.pkl"
if os.path.exists(model_result_path):
    with open(model_result_path, "rb") as f:
        model_parameters = pkl.load(f)
    batch_size = model_parameters["batch_size"]
    train_correlation = model_parameters["train_correlation"]
    test_correlation = model_parameters["test_correlation"]
    split_seed = model_parameters["split_seed"]
    shuffle_seed = model_parameters["shuffle_seed"]
    model_type = model_parameters["model_type"]
    optimizer_type = model_parameters["optimizer_type"]
else:
    print("Error ! You need to train the model before being able to test it !")
    raise FileExistsError(model_result_path)

result_path += f"test_model_{model_id}.pkl"

if os.path.exists(result_path):
    print(f"\n\n XXXXXXXX Testing model {model_id} skipped, already done XXXXXXXX")
else :
    print(f"XXXXXXXX Testing model {model_id} with parameters :")

    train_dataloader, validation_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=train_correlation, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed)
    test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False, shuffle_seed=shuffle_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = mu.get_model(model_type=model_type)
    model=model.to(device)
    assert(os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"))
    model.load_state_dict(torch.load(args.result_path + f"models/{model_name}/model_{model_id}"))

    loss_fn = nn.CrossEntropyLoss()

    adjacency_matrix = mu.adjacency_test(test_dataloader, model, loss_fn, device=device)

    with open(result_path, "wb") as f:
        pkl.dump(adjacency_matrix, f)
    
    print(f"OOOOOOOOO Testing model {model_id} runned with sucess OOOOOOOOO")

