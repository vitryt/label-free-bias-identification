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

import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Sequential(nn.Linear(28*28*3, 512), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.l2(x)
        logits = self.l3(x)
        return logits


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
    accuracy_matrix = correctness_matrix/appearance_matrix
    return accuracy_matrix


sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=int, default=0)
parser.add_argument("--exp_name", type=str, default="MNIST")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--train_correlation", type=float, default=0.7)
parser.add_argument("--test_correlation", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=50)

args = parser.parse_args()

exp_id = args.exp_id
exp_name = args.exp_name
batch_size = args.batch_size
train_correlation = args.train_correlation
test_correlation = args.test_correlation
epochs = args.epochs

result_path = "models/"
if not os.path.exists(result_path):
    os.mkdir(result_path)
result_path += exp_name + "/"
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_path += f"experiment_{exp_id}.pkl"
if os.path.exists(result_path):
    with open(result_path, "rb") as f:
        parameters = pkl.load(f)
    batch_size = parameters["batch_size"]
    train_correlation = parameters["train_correlation"]
    test_correlation = parameters["test_correlation"]
    epochs = parameters["epochs"]
else:
    parameters = {
        "exp_id":exp_id,
        "exp_name":exp_name,
        "batch_size":batch_size,
        "train_correlation":train_correlation,
        "test_correlation":test_correlation,
        "epochs":epochs,
    }

if os.path.exists(f"models/{exp_name}/model_{exp_id}"):
    print(f"\n\n XXXXXXXX Experiment number {exp_id} skipped, already done XXXXXXXX")

else :
    print(f"XXXXXXXX Starting experiment {exp_id} with parameters :", parameters)
    data_path = os.getcwd() + "/data/" + exp_name
    train_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=train_correlation, train=True)
    test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    accuracy_matrixes = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        print("                                            ")
        accuracy_matrix = test(test_dataloader, model, loss_fn)
    
    torch.save(model.state_dict(), f"models/{exp_name}/model_{exp_id}")

    parameters["accuracy_matrix"] = accuracy_matrix

    with open(result_path, "wb") as f:
        pkl.dump(parameters, f)
    
    print(f"OOOOOOOOO Experiment number {exp_id} runned with sucess OOOOOOOOO")

