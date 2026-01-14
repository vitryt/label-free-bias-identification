
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
from craft.craft_torch import Craft, torch_to_numpy
import src.model_utils as mu
import src.concept_utils as cu
from sklearn.decomposition import NMF

# import wandb

import numpy as np

import logging
logging.getLogger('tensorflow').disabled = True



sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")

parser.add_argument("--concept_id", type=str, default="0")
# parser.add_argument("--layer_depth", type=int, default=4)
# parser.add_argument("--number_of_concept", type=int, default=40)
# parser.add_argument("--patch_size", type=int, default=8)
# parser.add_argument("--concept_dataset_size", type=int, default=2000)
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
# layer_depth = args.layer_depth
# number_of_concept = args.number_of_concept
# patch_size = args.patch_size
# concept_dataset_size = args.concept_dataset_size
bias_labels = range(10)

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

bias_result_path = result_path + f"correlation_{model_id}_{concept_id}.pkl"
bias_parameters = {
}

if os.path.exists(bias_result_path):
    print(f"\n\n XXXXXXXX Correlation analysis experiment number {model_id}|{concept_id} skipped, already done XXXXXXXX")
else :
    print(f"XXXXXXXX Starting correlation analysis experiment {model_id}|{concept_id}")
    # run = wandb.init(
    #     entity="thomas-vitry",
    #     project="CVDB",
    #     config=parameters,
    # )

    # train_dataloader, validation_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=train_correlation, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed)
    # test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False, shuffle_seed=shuffle_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = (mu.CMNISTNeuralNetwork() if model_parameters["model_name"] == "CMNIST" else None).to(device)
    assert(os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"))
    model.load_state_dict(torch.load(args.result_path + f"models/{model_name}/model_{model_id}"))

    # loss_fn = nn.CrossEntropyLoss()

    g = nn.Sequential(*(list(model.children())[:layer_depth-1])) # Layers pre concept decomposition
    h = nn.Sequential(*(list(model.children())[layer_depth-1:])) # Layers post concept decompositon

    concept_engines = {}
    if not args.multi_concept:
        concept_engines["all"] = Craft(
            input_to_latent=g,
            latent_to_logit = h,
            number_of_concepts = number_of_concept,
            patch_size = patch_size,
            batch_size = batch_size,
            device = device
        )

        concept_engines["all"].reducer = concept_parameters["concept_parameters"]["reducer"]
        concept_engines["all"].W = np.array(concept_parameters["concept_parameters"]["W"], dtype=np.float32)
    else:
        for label in concept_parameters["concept_parameters"].keys():
            concept_engines[label] = Craft(
                input_to_latent=g,
                latent_to_logit=h,
                number_of_concepts=number_of_concept,
                patch_size=patch_size,
                batch_size=batch_size,
                device=device
            )
            concept_engines[label].reducer = concept_parameters["concept_parameters"][label]["reducer"]
            concept_engines[label].W = np.array(concept_parameters["concept_parameters"][label]["W"], dtype=np.float32)

    test_dataloader = get_biased_mnist_dataloader(root=data_path, batch_size=batch_size, data_label_correlation=test_correlation, train=False, shuffle_seed=shuffle_seed)
    res = {}
    for X, y, y_pred in test_dataloader:
        X_activation = list(concept_engines.values())[0].input_to_latent(X.to(device)).cpu()
        for bias_decomposer_label, concept_engine in concept_engines.items():
            if not bias_decomposer_label in res:
                res[bias_decomposer_label] = {}
            X_concepts = concept_engine.transform(inputs=None, activations=X_activation)
            for bias_label in y_pred.unique():
                bias_label = int(bias_label)
                if bias_label in res[bias_decomposer_label]:
                    res[bias_decomposer_label][bias_label] = np.concatenate([res[bias_decomposer_label][bias_label], X_concepts[y_pred == bias_label]])
                else:
                    res[bias_decomposer_label][bias_label] = X_concepts[y_pred==bias_label]
    
    with open(bias_result_path, "wb") as f:
        pkl.dump(res, f)
    
    # run.finish()
    print(f"OOOOOOOOO Correlation analysis experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

