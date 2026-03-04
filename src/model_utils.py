# Tools to make a classification model on CMNIST
from torch import nn
import numpy as np
import torch

def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    """
    General training loop
    
    Expecting batches of at least (X, y). 
    """
    size = len(dataloader.dataset)
    model.train()

    for batch, batch_data in enumerate(dataloader):
        if isinstance(batch_data, (list, tuple)):
            X = batch_data[0]
            y = batch_data[1]
        else:
            raise ValueError(f"Unexpected batch type received in training loop ! {type(batch_data)}")
        
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="\r")


def test(dataloader, model, loss_fn, device="cpu"):
    """
    Evaluation loop
    
    Expects batches of:
    * (X, y) or
    * (X, y, bias_label/attribute) or
    * (X, y, bias_label, group) 

    In the case a third element is passed to the function, it is treated as the bias or psuedo-group label
    which in CMNIST was originally y_pred. If no third element is passed, the function falls back to y itself 
    to ensure the matrices consistent.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    appearance_matrix = []
    correctness_matrix = None
    with torch.no_grad():
        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                X = batch_data[0]
                y = batch_data[1]
                if len(batch_data) >= 3:
                    y_pred = batch_data[2]
                else:
                    y_pred = y
            else:
                raise ValueError(f"Unexpected batch type received in test loop ! {type(batch_data)}")

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
    # run.log({"acc" : correct, "loss":test_loss})

    return correctness_matrix, appearance_matrix

def adjacency_test(dataloader, model, loss_fn, device="cpu"):
    # Does not work with dataset where the bias is not given
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    adjacency_matrix = []
    with torch.no_grad():
        for X, y, y_pred in dataloader:
            X, y, y_pred = X.to(device), y.to(device), y_pred.to(device)
            pred = model(X)
            pred_arg = pred.argmax(1)
            if len(adjacency_matrix)==0:
                size_y = len(pred[0])
                adjacency_matrix = np.array([[0 for i in range(size_y)] for j in range(size_y)])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i in range(size_y):
                for j in range(size_y):
                    adjacency_matrix[i][j] += int(((pred_arg==i) & (y_pred==j)).sum())
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # run.log({"acc" : correct, "loss":test_loss})
    return adjacency_matrix


# Model imports --> we could move to conditional statements
from src.colour_mnist import CMNISTNeuralNetwork
from src.waterbird import WaterbirdsResNet18, WaterbirdsResNet50
from src.celeba import CelebAResNet50
from src.urbancars import UrbanCarsResNet50, UrbanCarsResNet18

def get_model(model_type):
    if model_type == "MLP":
        return CMNISTNeuralNetwork()
    elif model_type == "resnet18":
        return WaterbirdsResNet18(num_classes = 2)
    elif model_type == "resnet50":
        return WaterbirdsResNet50(num_classes = 2)
    elif model_type == "resnetceleb":
        return CelebAResNet50()
    elif model_type == "resneturban50":
        return UrbanCarsResNet50()
    elif model_type == "resneturban18":
        return UrbanCarsResNet18()
    else:
        raise ValueError("Not a defined model being specified in model_training.py!")


def get_optimizer(optimizer_type, model):
    if optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr = 1e-3)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr = 1e-3)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr = 1e-3)
    else:
        raise ValueError("Not a defined optimiser being specified in model_training.py!")