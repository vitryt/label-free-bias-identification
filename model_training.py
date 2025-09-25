import argparse
import sys
import pickle as pkl
import os
import random

import torch
from src.colour_mnist import get_biased_mnist_dataloader
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from craft.craft_torch import Craft, torch_to_numpy
import src.model_utils as mu

import wandb

import numpy as np


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    appearance_matrix = []
    correctness_matrix = None
    with torch.no_grad():
        for X, y, y_pred in dataloader:
            X, y, y_pred = X.to(device), y.to(device), y_pred.to(device)
            pred = model(X)
            if len(appearance_matrix)==0:
                size_y = len(pred[0])
                appearance_matrix = np.array([[0 for i in range(size_y)] for j in range(size_y)])
                correctness_matrix = np.array([[0 for i in range(size_y)] for j in range(size_y)])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i in range(size_y):
                for j in range(size_y):
                    appearance_matrix[i][j] += int(((y==i) & (y_pred==j)).sum())
                    correctness_matrix[i][j] += int((pred.argmax(1)[((y==i) & (y_pred==j))] == i).sum())
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    run.log({"acc" : correct, "loss":test_loss})
    return correctness_matrix, appearance_matrix


sys.path.append(os.getcwd())

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

args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name
batch_size = args.batch_size
train_correlation = args.train_correlation
test_correlation = args.test_correlation
epochs = args.epochs
split_seed = args.split_seed
shuffle_seed = args.shuffle_seed

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
    }

if os.path.exists(f"models/{model_name}/model_{model_id}"):
    print(f"\n\n XXXXXXXX Training of model {model_id} skipped, already done XXXXXXXX")

else :
    print(f"XXXXXXXX Starting training of model {model_id} with parameters :", parameters)
    run = wandb.init(
        entity="thomas-vitry",
        project="CVDB",
        config=parameters,
    )

    data_path = os.getcwd() + "/data/" + model_name
    train_dataloader, validation_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=train_correlation, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed)
    test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False, shuffle_seed=shuffle_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = (mu.CMNISTNeuralNetwork() if parameters["model_name"] == "MNIST" else None).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    accuracy_matrixes = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("                                            ")
        correctness_matrix, appearance_matrix = test(test_dataloader, model, loss_fn)
    
    torch.save(model.state_dict(), f"models/{model_name}/model_{model_id}")

    parameters["correctness_matrix"] = correctness_matrix
    parameters["appearance_matrix"] = appearance_matrix

    with open(result_path, "wb") as f:
        pkl.dump(parameters, f)
    
    run.finish()
    print(f"OOOOOOOOO Training of model number {model_id} runned with sucess OOOOOOOOO")

