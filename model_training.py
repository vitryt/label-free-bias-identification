import argparse
import sys
import pickle as pkl
import os
import random
import wandb
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import src.model_utils as mu

# Dataset imports --> we could move to conditional statements
from src.colour_mnist import get_biased_mnist_dataloader # Coloured MNIST
from src.waterbird import WaterBirdsDataset # Waterbirds

# Model imports --> we could move to conditional statements
from src.resnet18_utils import WaterbirdsResNet18
from src.resnet50_utils import WaterbirdsResNet50


sys.path.append(os.getcwd())

# TODO Kieran add arguments for the dataset, model and optimizer
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_correlation", type=float, default=0.8)
parser.add_argument("--test_correlation", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument("--shuffle_seed", type=int, default=42)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")

# Updated arguments for introduction of Waterbirds dataset, ResNet models and modularity
parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "Waterbirds"]) # Add CelebA down the line
parser.add_argument("--model_type", type=str, default="MLP", choices=["MLP", "resnet18", "resnet50"])
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])

args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name
batch_size = args.batch_size
train_correlation = args.train_correlation
test_correlation = args.test_correlation
epochs = args.epochs
split_seed = args.split_seed
shuffle_seed = args.shuffle_seed


dataset = args.dataset
model_type = args.model_type
optimizer_type = args.optimizer

data_path = args.data_path
if data_path == "":
    data_path = os.getcwd()
data_path += "/data/" + dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

result_path = args.result_path + "models/"
if not os.path.exists(result_path):
    os.mkdir(result_path)
result_path += model_name + "/"
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_path += f"model_{model_id}.pkl"
if os.path.exists(result_path):
    with open(result_path, "rb") as f:
        parameters = pkl.load(f)
    # TODO add the new parameters here
    batch_size = parameters["batch_size"]
    train_correlation = parameters["train_correlation"]
    test_correlation = parameters["test_correlation"]
    epochs = parameters["epochs"]
    split_seed = parameters["split_seed"]
    shuffle_seed = parameters["shuffle_seed"]
    dataset = parameters.get("dataset", "MNIST")
    model_type = parameters.get("model_type", "MLP")
    optimizer_type = parameters.get("optimizer", "sgd")
else:
    parameters = { # TODO Kieran add the new parameters here
        "model_id":model_id,
        "model_name":model_name,
        "batch_size":batch_size,
        "train_correlation":train_correlation,
        "test_correlation":test_correlation,
        "epochs":epochs,
        "split_seed":split_seed,
        "shuffle_seed":shuffle_seed,
        "dataset": args.dataset,
        "model_type": args.model_type,
        "optimizer": args.optimizer,
    }

if os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"):
    print(f"\n\n XXXXXXXX Training of model {model_id} skipped, already done XXXXXXXX")

else :
    print(f"XXXXXXXX Starting training of model {model_id} with parameters :", parameters)
    # run = wandb.init(
    #     entity="thomas-vitry",
    #     project="CVDB",
    #     config=parameters,
    # )

    # TODO Change so that the datasets are a parameter
    # train_dataloader, validation_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=train_correlation, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed)
    # test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False, shuffle_seed=shuffle_seed)
    test_func = mu.adjacency_test
    if args.dataset == "MNIST":
        train_dataloader, validation_dataloader = get_biased_mnist_dataloader(
            root = data_path, 
            batch_size = batch_size, 
            data_label_correlation = train_correlation,
            train = True, 
            validation = 1/10, 
            split_gen_seed = split_seed, 
            shuffle_seed = shuffle_seed
        )
        test_dataloader = get_biased_mnist_dataloader(
            root = data_path, 
            batch_size = batch_size, 
            data_label_correlation = test_correlation,
            train = False, 
            shuffle_seed = shuffle_seed
        )
        test_func = mu.test
    elif args.dataset == "Waterbirds":
        transform = None  # Using default
        train_dataloader = DataLoader(WaterBirdsDataset(data_path, "train", transform), batch_size = batch_size, shuffle = True)
        validation_dataloader = DataLoader(WaterBirdsDataset(data_path, "val", transform), batch_size = batch_size)
        test_dataloader = DataLoader(WaterBirdsDataset(data_path, "test", transform), batch_size = batch_size)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # model = (mu.CMNISTNeuralNetwork() if parameters["model_name"] == "MNIST" else None).to(device) # TODO Change so that the model is a parameter

    if args.model_type == "MLP":
        model = mu.CMNISTNeuralNetwork().to(device)
    elif args.model_type == "resnet18":
        model = WaterbirdsResNet18(num_classes = 2).to(device)
    elif args.model_type == "resnet50":
        model = WaterbirdsResNet50(num_classes = 2).to(device)
    else:
        raise ValueError("Not a defined model being specified in model_training.py!")
        
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # TODO Change this so that the optimizer is a parameter
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    else:
        raise ValueError("Not a defined optimiser being specified in model_training.py!")

    accuracy_matrixes = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        mu.train(train_dataloader, model, loss_fn, optimizer, device=device)
        print("                                            ")
        adjacency_matrix = test_func(test_dataloader, model, loss_fn, device=device) # TODO change to use a different test function in case you modified that
    
    torch.save(model.state_dict(), args.result_path + f"models/{model_name}/model_{model_id}")

    parameters["result_matrix"] = adjacency_matrix

    with open(result_path, "wb") as f:
        pkl.dump(parameters, f)
    
    # run.finish()
    print(f"OOOOOOOOO Training of model number {model_id} runned with sucess OOOOOOOOO")

