
import argparse
import sys
import pickle as pkl
import os
import random

import torch
from src.colour_mnist import get_bias_difference_mnist_dataloader
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from craft.craft_torch import Craft, torch_to_numpy
import src.model_utils as mu
import src.concept_utils as cu
from sklearn.decomposition import NMF

# import wandb

import numpy as np


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

parser.add_argument("--concept_id", type=int, default=0)
# parser.add_argument("--layer_depth", type=int, default=4)
# parser.add_argument("--number_of_concept", type=int, default=40)
# parser.add_argument("--patch_size", type=int, default=8)
# parser.add_argument("--concept_dataset_size", type=int, default=2000)
# parser.add_argument("--backprop_step", type=int, default=1000)
# parser.add_argument("--concept_threshold", type=float, default=0.3)


args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name

concept_id = args.concept_id
# layer_depth = args.layer_depth
# number_of_concept = args.number_of_concept
# patch_size = args.patch_size
# concept_dataset_size = args.concept_dataset_size
bias_labels = range(10)

result_path = "models/"
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
else:
    print("Error ! You need to train the model before being able to test it !")
    raise FileExistsError(model_result_path)

concept_result_path = result_path + f"concepts_{model_id}_{concept_id}.pkl"
if os.path.exists(concept_result_path):
    with open(concept_result_path, "rb") as f:
        concept_parameters = pkl.load(f)
    layer_depth = concept_parameters["layer_depth"]
    number_of_concept = concept_parameters["number_of_concept"]
    patch_size = concept_parameters["patch_size"]
    concept_dataset_size = concept_parameters["concept_dataset_size"]
else:
    print("Error ! You need to build the concepts before being able to test them !")
    raise FileExistsError(concept_result_path)

bias_result_path = result_path + f"bias_{model_id}_{concept_id}.pkl"
bias_parameters = {
}

if os.path.exists(bias_result_path):
    print(f"\n\n XXXXXXXX Bias analysis experiment number {model_id}|{concept_id} skipped, already done XXXXXXXX")
else :
    print(f"XXXXXXXX Starting bias analysis experiment {model_id}|{concept_id}")
    # run = wandb.init(
    #     entity="thomas-vitry",
    #     project="CVDB",
    #     config=parameters,
    # )

    data_path = os.getcwd() + "/data/" + model_name
    # train_dataloader, validation_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=train_correlation, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed)
    # test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False, shuffle_seed=shuffle_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = (mu.CMNISTNeuralNetwork() if model_parameters["model_name"] == "MNIST" else None).to(device)
    assert(os.path.exists(f"models/{model_name}/model_{model_id}"))
    model.load_state_dict(torch.load(f"models/{model_name}/model_{model_id}"))

    # loss_fn = nn.CrossEntropyLoss()

    g = nn.Sequential(*(list(model.children())[:layer_depth-1])) # Layers pre concept decomposition
    h = nn.Sequential(*(list(model.children())[layer_depth-1:])) # Layers post concept decompositon

    craft = Craft(
        input_to_latent=g,
        latent_to_logit = h,
        number_of_concepts = number_of_concept,
        patch_size = patch_size,
        batch_size = batch_size,
        device = device
    )

    craft.reducer = concept_parameters["reducer"]
    craft.W = np.array(concept_parameters["W"], dtype=np.float32)

    bias_dataloaders = {
        bias_label: get_bias_difference_mnist_dataloader(root=data_path, batch_size=batch_size, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed, bias_colour=bias_label)[1]
        for bias_label in bias_labels}

    results = cu.gather_all_bias_results(bias_dataloaders, model, craft, device=device)
    
    with open(bias_result_path, "wb") as f:
        pkl.dump(results, f)
    
    # run.finish()
    print(f"OOOOOOOOO Bias analysis experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

