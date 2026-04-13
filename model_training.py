import argparse
import sys
import pickle as pkl
import os
import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import src.model_utils as mu
import src.dataset_utils as du



sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_correlation", type=float, default=0.95)
parser.add_argument("--test_correlation", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument("--shuffle_seed", type=int, default=42)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")

# Updated arguments for introduction of Waterbirds dataset, ResNet models and modularity
parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "Waterbirds", "CelebA"])
parser.add_argument("--model_type", type=str, default="MLP", choices=["cnn", "MLP", "resnet18", "resnet50", "resnetceleb"])
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "adamw"])
parser.add_argument("--starting_lr", type=float, default=0.01)
parser.add_argument("--lr_decrease_factor", type=float, default=0.5)
parser.add_argument("--lr_decrease_frequency", type=int, default=25)

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
starting_lr = args.starting_lr
lr_decrease_factor = args.lr_decrease_factor
lr_decrease_frequency = args.lr_decrease_frequency

data_path = args.data_path
if data_path == "":
    data_path = os.getcwd()
data_path += "/" + dataset

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
    batch_size = parameters["batch_size"]
    train_correlation = parameters["train_correlation"]
    test_correlation = parameters["test_correlation"]
    epochs = parameters["epochs"]
    split_seed = parameters["split_seed"]
    shuffle_seed = parameters["shuffle_seed"]
    dataset = parameters["dataset"]
    model_type = parameters["model_type"]
    optimizer_type = parameters["optimizer_type"]
    starting_lr = parameters.get("starting_lr", starting_lr)
    lr_decrease_factor = parameters.get("lr_decrease_factor", lr_decrease_factor)
    lr_decrease_frequency = parameters.get("lr_decrease_frequency", lr_decrease_frequency)
else:
    parameters = {
        "model_id":model_id,
        "model_name":model_name,
        "batch_size":batch_size,
        "train_correlation":train_correlation,
        "test_correlation":test_correlation,
        "epochs":epochs,
        "split_seed":split_seed,
        "shuffle_seed":shuffle_seed,
        "dataset":dataset,
        "model_type": model_type,
        "optimizer_type":optimizer_type,
        "starting_lr":starting_lr,
        "lr_decrease_factor":lr_decrease_factor,
        "lr_decrease_frequency":lr_decrease_frequency,
    }

if os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"):
    print(f"\n\n XXXXXXXX Training of model {model_id} skipped,  already done XXXXXXXX")

else :
    print(f"XXXXXXXX Starting training of model {model_id} with parameters :", parameters)
    test_func = mu.test
    train_dataloader, test_dataloader = du.get_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        split=["train", "test"],
        train_correlation=train_correlation,
        test_correlation=test_correlation,
        batch_size=batch_size,
        seeds=(split_seed, shuffle_seed)
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = mu.get_model(model_type=model_type)
    model=model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = mu.get_optimizer(optimizer_type=optimizer_type, model=model, lr=starting_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_decrease_frequency,
        gamma=lr_decrease_factor,
    )

    accuracy_matrixes = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        mu.train(train_dataloader, model, loss_fn, optimizer, device=device)
        print("                                            ")
        correctness_matrix, appearance_matrix = mu.test(test_dataloader, model, loss_fn, device=device)
        scheduler.step()
        print(f"Learning rate after epoch {t+1} : {optimizer.param_groups[0]['lr']}")
    torch.save(model.state_dict(), args.result_path + f"models/{model_name}/model_{model_id}")
    parameters["correctness_matrix"] = correctness_matrix
    parameters["appearance_matrix"] = appearance_matrix

    with open(result_path, "wb") as f:
        pkl.dump(parameters, f)
    
    print(f"OOOOOOOOO Training of model number {model_id} runned with sucess OOOOOOOOO")

