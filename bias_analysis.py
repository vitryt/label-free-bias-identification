import argparse
import sys
import pickle as pkl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import logging

import torch
from torch import nn
from torchvision.transforms import ToTensor
from craft.craft_torch import Craft, torch_to_numpy
import src.model_utils as mu
import src.concept_utils as cu
import src.dataset_utils as du

from src.colour_mnist import get_bias_difference_mnist_dataloader
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

logging.getLogger('tensorflow').disabled = True

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")

parser.add_argument("--experiment_name", type=str, default="experiment")
parser.add_argument("--layer_depth", type=int, default=4)
parser.add_argument("--number_of_concept", type=int, default=40)
parser.add_argument("--patch_size", type=int, default=8)
# parser.add_argument("--backprop_step", type=int, default=1000)
# parser.add_argument("--concept_threshold", type=float, default=0.3)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--val_correlation", type=float, default=-1)


args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name

experiment_name = args.experiment_name
layer_depth = args.layer_depth
number_of_concept = args.number_of_concept
patch_size = args.patch_size
concept_id = f"{experiment_name}_{number_of_concept}_{patch_size}"

data_path = args.data_path
if data_path == "":
    data_path = os.getcwd()

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
    dataset = model_parameters["dataset"]
    model_type = model_parameters["model_type"]
else:
    print("Error ! You need to train the model before being able to test it !")
    raise FileExistsError(model_result_path)

data_path += "/" + dataset

concept_result_path = result_path + f"concepts_{model_id}_{concept_id}.pkl"
if os.path.exists(concept_result_path):
    with open(concept_result_path, "rb") as f:
        concept_parameters = pkl.load(f)
    layer_depth = concept_parameters["layer_depth"]
    number_of_concept = concept_parameters["number_of_concept"]
    patch_size = concept_parameters["patch_size"]
    concept_dataset_size = concept_parameters.get("concept_dataset_size")
else:
    print("Error ! You need to build the concepts before being able to test them !")
    raise FileExistsError(concept_result_path)

if "bias_alignment_values" in concept_parameters:
    print(f"\n\n XXXXXXXX Bias analysis experiment number {model_id}|{concept_id} skipped, already done XXXXXXXX")
else :
    print(f"XXXXXXXX Starting bias analysis experiment {model_id}|{concept_id}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = mu.get_model(model_type=model_type)
    model=model.to(device)
    assert(os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"))
    model.load_state_dict(torch.load(args.result_path + f"models/{model_name}/model_{model_id}"))

    g, h = mu._split_model(model, args.layer_depth)

    concept_engines = {}
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

    bias_dataloaders = {
        bias_label: get_bias_difference_mnist_dataloader(root=data_path, batch_size=batch_size, train=True, validation=1/10, split_gen_seed=split_seed, shuffle_seed=shuffle_seed, bias_colour=bias_label)[1]
        for bias_label in range(10)}

    bias_res = cu.gather_all_bias_results(bias_dataloaders, model, concept_engines, device=device)
    results = []
    for studied_bias in range(10):
        W = concept_parameters["concept_parameters"][studied_bias]["W"]
        tmp = []
        amount = 0
        for label in bias_res[studied_bias]:
            tmp.append(bias_res[studied_bias][label][2])
            # amount += len(bias_res[studied_bias][label][2])
        bias_vector = np.concatenate(tmp, axis=0)
        bias_vector = bias_vector.mean(axis=0)
        alignment = cos_sim([bias_vector], W)[0]
        results.append(alignment)

    concept_parameters["bias_alignment_values"] = results

    with open(concept_result_path, "wb") as f:
        pkl.dump(concept_parameters, f)
    
    # run.finish()
    print(f"OOOOOOOOO Bias analysis experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

