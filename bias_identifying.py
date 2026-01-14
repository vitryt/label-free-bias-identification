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
backprop_steps = [100, 300, 700, 1000]

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

data_path += "/data/" + dataset

result_path += f"concepts_{model_id}_{concept_id}.pkl"
parameters = {
    "model_id":model_id,
    "concept_id":concept_id,
    "layer_depth":layer_depth,
    "number_of_concept":number_of_concept,
    "patch_size":patch_size,
    "concept_dataset_size":concept_dataset_size,
    "backprop_steps":backprop_steps,

}

if os.path.exists(result_path):
    print(f"\n\n XXXXXXXX Concept experiment number {model_id}|{concept_id} skipped, already done XXXXXXXX")
else :
    print(f"XXXXXXXX Starting concept experiment {model_id}|{concept_id} with parameters :", parameters)
    # run = wandb.init(
    #     entity="thomas-vitry",
    #     project="CVDB",
    #     config=parameters,
    # )

    # Specifying dataset loader
    if dataset == "MNIST":
        training_dataloader, validation_dataloader = get_biased_mnist_dataloader(
            root = data_path,
            batch_size = batch_size,
            data_label_correlation = 0.1,#model_parameters["train_correlation"],
            train = True,
            validation = 1/10,
            split_gen_seed = split_seed,
            shuffle_seed = shuffle_seed
        )
    if dataset == "Waterbirds":
        training_data = WaterBirdsDataset(
            root_dir = data_path,
            split = "train"
        )
        validation_data = WaterBirdsDataset(
            root_dir = data_path,
            split = "val"
        )
        training_dataloader = DataLoader(
            training_data,
            batch_size = batch_size,
            shuffle = True
        )
        validation_dataloader = DataLoader(
            validation_data,
            batch_size = batch_size,
            shuffle = False
        )



    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if model_type == "MLP":
        model = mu.CMNISTNeuralNetwork()
    elif model_type == "resnet18":
        model = WaterbirdsResNet18()
    elif model_type == "resnet50":
        model = WaterbirdsResNet50()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    assert(os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"))
    model.load_state_dict(torch.load(args.result_path + f"models/{model_name}/model_{model_id}"))

    loss_fn = nn.CrossEntropyLoss()
    concepts_engines, concept_parameters = cu.train_concept_engines(
        dataloader=validation_dataloader,
        model=model,
        layer_depth=layer_depth,
        number_of_concept=number_of_concept,
        concept_decomposer=Craft,
        patch_size=patch_size,
        concept_dataset_size=concept_dataset_size,
        batch_size=batch_size,
        device=device,
        get_masks=True if dataset=="Waterbirds" else False
    )
    print("-Concept decomposition trained successfully")
    gr = cu.Gradient_retriever(model.backbone.fc if hasattr(model, "backbone") else list(model.children())[layer_depth-1])

    results = {}
    for backprop_step in backprop_steps:
        print(f"-Gathering concepts with backprop step : {backprop_step}", end="\r")
        results[backprop_step] = cu.gather_all_concept_results(validation_dataloader, model, loss_fn, concepts_engines, gr, backprop_mult=backprop_step, device=device, multi_concept=args.multi_concept)
        print(f"-Concepts with backprop step {backprop_step} gathered successfully")

        # res = {}
        # for classe in range(10):
        #     biases = cu.get_bias_concept(results=results, studied_class=classe, number_of_concepts=number_of_concept)
        #     study_list = list(biases.items())
        #     study_list.sort(key=lambda x : x[1])
        #     for c_id, val in study_list[::-1][:]:
        #         if c_id in res:
        #             res[c_id] = max(res[c_id], val)
        #         else :
        #             res[c_id] = val
        # res = [(key, val) for key, val in res.items()]
        # res.sort(key= lambda x: -x[1])

    # parameters["concept_bias"] = res
    parameters["concept_results"] = results
    parameters["concept_parameters"] = concept_parameters

    with open(result_path, "wb") as f:
        pkl.dump(parameters, f)
    
    # run.finish()
    print(f"OOOOOOOOO Concept experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

