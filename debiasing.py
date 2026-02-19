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
from craft.craft_torch import Craft
import src.model_utils as mu
import src.concept_utils as cu
import src.dataset_utils as du


import src.nmf_merge as nmergef
from sklearn.decomposition import NMF


logging.getLogger('tensorflow').disabled = True

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=int, default=0)
parser.add_argument("--model_name", type=str, default="MNIST")

parser.add_argument("--concept_id", type=str, default="0_0")
# parser.add_argument("--backprop_step", type=int, default=1000)
# parser.add_argument("--concept_threshold", type=float, default=0.3)
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")

parser.add_argument("--bias_threshold", type=int, default=65)
parser.add_argument("--backprop_step", type=int, default=300)


args = parser.parse_args()

model_id = args.model_id
model_name = args.model_name

concept_id = args.concept_id

bias_threshold = args.bias_threshold /100
backprop_step = args.backprop_step

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

concept_result_path = result_path + f"concepts_{model_id}_{concept_id}.pkl"
if os.path.exists(concept_result_path):
    with open(concept_result_path, "rb") as f:
        concept_parameters = pkl.load(f)
    layer_depth = concept_parameters["layer_depth"]
    number_of_concept = concept_parameters["number_of_concept"]
    patch_size = concept_parameters["patch_size"]
    concept_dataset_size = concept_parameters["concept_dataset_size"]
    res_backprop_steps = concept_parameters["backprop_steps"]
else:
    print("Error ! You need to build the concepts before being able to test them !")
    raise FileExistsError(concept_result_path)


if not backprop_step in res_backprop_steps:
    raise ValueError(f"Decomposition was not built for backprop steps {backprop_step}. List of valid setps : {res_backprop_steps}")

debias_result_path = result_path + f"debias_{model_id}_{concept_id}.pkl"
if not os.path.exists(debias_result_path):
    concept_res = concept_parameters["concept_results"]

    debias_results = {"bias_vectors": [], "bias_threshold": bias_threshold}

    for label in concept_parameters["concept_parameters"].keys():
        # concept_engines[label] = Craft(
        #     input_to_latent=g,
        #     latent_to_logit=h,
        #     number_of_concepts=number_of_concept,
        #     patch_size=patch_size,
        #     batch_size=batch_size,
        #     device=device
        # )
        # concept_engines[label].reducer = concept_parameters["concept_parameters"][label]["reducer"]
        # concept_engines[label].W = np.array(concept_parameters["concept_parameters"][label]["W"], dtype=np.float32)
        if "false_n" in concept_res[backprop_step][label]:
            results = concept_res[backprop_step][label]["false_n"]
            appearance = len(results["concept_base"])
            base_freq = ((results["concept_base"] > 0).sum(axis=0))/appearance
            ascent_freq = (results["concept_ascent"] > 0).sum(axis=0)/appearance
            descent_freq = (results["concept_descent"] > 0).sum(axis=0)/appearance
        else:
            base_freq = np.array([0 for i in range(number_of_concept)])
            ascent_freq = np.array([0 for i in range(number_of_concept)])
            descent_freq = np.array([0 for i in range(number_of_concept)])
        false_n_res = descent_freq - base_freq
        false_n_res[false_n_res < 0] = 0
        if "false_p" in concept_res[backprop_step][label]:
            results = concept_res[backprop_step][label]["false_p"]
            appearance = len(results["concept_base"])
            base_freq = ((results["concept_base"] > 0).sum(axis=0))/appearance
            ascent_freq = (results["concept_ascent"] > 0).sum(axis=0)/appearance
            descent_freq = (results["concept_descent"] > 0).sum(axis=0)/appearance
        else: 
            base_freq = np.array([0 for i in range(number_of_concept)])
            ascent_freq = np.array([0 for i in range(number_of_concept)])
            descent_freq = np.array([0 for i in range(number_of_concept)])
        false_p_res = base_freq - descent_freq
        false_p_res[false_p_res < 0] = 0

        final_res = (false_p_res + false_n_res) /2
        debias_results[label]=final_res
        if final_res.max() > bias_threshold:
            debias_results["bias_vectors"].append((label, final_res.argmax()))

    print("-Concept decomposition loaded successfully")

    Wmerged, cluster_labels = nmergef.component_level_fusion([concept_parameters["concept_parameters"][label]["W"] for label in concept_parameters["concept_parameters"].keys()], similarity_threshold=0.95)
    cluster_labels -= 1
    bias_labels = []
    for decomposition, rank in debias_results["bias_vectors"]:
        cluster_label = cluster_labels[decomposition * number_of_concept + rank]
        if not cluster_label in bias_labels:
            bias_labels.append(cluster_label)
    bias_labels.sort()
    debias_results["Wmerged"] = Wmerged
    debias_results["bias_merged"] = bias_labels
    with open(debias_result_path, "wb") as f:
        pkl.dump(debias_results, f)
    print("New merged decomposition built and saved")
else:
    print("Pre-study found and loaded")

with open(debias_result_path, "rb") as f:
    debias_results = pkl.load(f)

# Specifying dataset loader
test_dataloader = du.get_dataloaders(
    dataset_name=dataset,
    data_path=data_path,
    split=["test"],
    train_correlation=train_correlation,
    test_correlation=test_correlation,
    batch_size=batch_size,
    seeds=(split_seed, shuffle_seed)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = mu.get_model(model_type=model_type)
model = model.to(device)
assert(os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"))
model.load_state_dict(torch.load(args.result_path + f"models/{model_name}/model_{model_id}"))

loss_fn = nn.CrossEntropyLoss()
if hasattr(model, "backbone"):
    backbone = model.backbone
    g = nn.Sequential(*(list(model.backbone.children())[:-1] + [nn.Flatten(start_dim=1)])) # Layers pre concept decomposition
    h = nn.Sequential(backbone.fc) # Layers post concept decompositon
else:
    g = nn.Sequential(*(list(model.children())[:layer_depth-1])) # Layers pre concept decomposition (CMNIST version)
    h = nn.Sequential(*(list(model.children())[layer_depth-1:])) # Layers post concept decompositon (CMNIST version)



craft = Craft(
    input_to_latent=g,
    latent_to_logit=h,
    number_of_concepts=debias_results["Wmerged"].shape[0],
    patch_size=patch_size,
    batch_size=batch_size,
    device=device
)
reducer = NMF(n_components=debias_results["Wmerged"].shape[0], max_iter = 10000)
reducer.components_ = debias_results["Wmerged"]
craft.reducer = reducer
craft.W = np.array(debias_results["Wmerged"], dtype=np.float32)

class Activation_disturber():
    def __init__(self,  layer, concepts, craft, device = "cuda"):
        self.craft_decomposer = craft
        self.bias_concepts = concepts
        self.handle = layer.register_forward_hook(self._hook_function)
        self.device = device
    
    def _hook_function(self, model, input, output):
        concepts_activation = craft.transform(inputs=None, activations=output)
        res = output - torch.Tensor(concepts_activation[:,self.bias_concepts] @ self.craft_decomposer.W[self.bias_concepts]).to(device)
        # res = output + torch.Tensor(0.05 * np.ones_like(concepts_activation[:,self.concepts]) @ self.cr.W[self.concepts]).to(device)
        return res

    def __del__(self):
        self.handle.remove()

ad = Activation_disturber(layer = g[-1], concepts=debias_results["bias_merged"], craft=craft, device=device)
n_model = nn.Sequential(g,h)
debias_results["correctness_matrix"], debias_results["appearance_matrix"] = mu.test(test_dataloader, n_model, loss_fn, device=device)

with open(debias_result_path, "wb") as f:
    pkl.dump(debias_results, f)

# run.finish()
print(f"OOOOOOOOO Debiasing experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

