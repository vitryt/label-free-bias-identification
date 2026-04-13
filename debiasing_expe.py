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
import src.nmf_merge as nmergef

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

parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--result_path", type=str, default="")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--val_correlation", type=float, default=-1)

parser.add_argument("--backprop_step", type=int, default=1000)
parser.add_argument("--concept_threshold", type=float, default=0.3)
parser.add_argument("--overwrite_phase", type=int, nargs="+", default=[])

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

debiasing_path = result_path + f"debiasing_{model_id}_{concept_id}.pkl"
if os.path.exists(debiasing_path):
    with open(debiasing_path, "rb") as f:
        debiasing_res = pkl.load(f)
    print(f"\n\n XXXXXXXX Debiasing analysis experiment number {model_id}|{concept_id} already done, loaded XXXXXXXX")
else :
    debiasing_res = {}



print(f"XXXXXXXX Starting debiasing analysis experiment {model_id}|{concept_id}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = mu.get_model(model_type=model_type)
model=model.to(device)
assert(os.path.exists(args.result_path + f"models/{model_name}/model_{model_id}"))
model.load_state_dict(torch.load(args.result_path + f"models/{model_name}/model_{model_id}"))

g, h = mu._split_model(model, args.layer_depth)

# Phase 1: compute the bias estimator value
if "bias_estimator" not in debiasing_res or 1 in args.overwrite_phase:
    print("Phase 1: computing the bias estimator value")
    debiasing_res["bias_estimator"] = cu.get_bias_estimator(data_folder=args.result_path, exp_name=model_name, exp_id=model_id, exp_type=experiment_name, concept_id=number_of_concept, patch_id=patch_size, backprop_step=args.backprop_step, bias_threshold=args.concept_threshold, mode="paper")
    with open(debiasing_path, "wb") as f:
        pkl.dump(debiasing_res, f)

# Phase 2: merge the concept decompositions
if "Wmerged" not in debiasing_res or 2 in args.overwrite_phase:
    print("Phase 2: merging the concept decompositions")
    Wmerged, cluster_labels = nmergef.component_level_fusion([concept_parameters["concept_parameters"][label]["W"] for label in concept_parameters["concept_parameters"].keys()], similarity_threshold=0.95)
    cluster_labels -= 1
    bias_labels = []
    for decomposition, rank in debiasing_res["bias_estimator"]["bias_vectors"]:
        cluster_label = cluster_labels[decomposition * number_of_concept + rank]
        if not cluster_label in bias_labels:
            bias_labels.append(cluster_label)
    bias_labels.sort()
    debiasing_res["Wmerged"] = Wmerged
    debiasing_res["cluster_labels"] = cluster_labels
    debiasing_res["bias_merged"] = bias_labels
    with open(debiasing_path, "wb") as f:
        pkl.dump(debiasing_res, f)

# Interphase, load the merged concept decomposition
craft = Craft(
    input_to_latent=g,
    latent_to_logit=h,
    number_of_concepts=debiasing_res["Wmerged"].shape[0],
    patch_size=patch_size,
    batch_size=batch_size,
    device=device
)
reducer = NMF(n_components=debiasing_res["Wmerged"].shape[0], max_iter = 10000)
reducer.components_ = debiasing_res["Wmerged"]
craft.reducer = reducer
craft.W = np.array(debiasing_res["Wmerged"], dtype=np.float32)
print("-" * 30)
print(f"Bias merged concept labels : {debiasing_res['bias_merged']}")
print("-" * 30)

# Phase 3: Chi squared test on the bias concept presence according to the target class
if "chi2_test" not in debiasing_res or 3 in args.overwrite_phase:
    print("Phase 3: Chi squared test on the bias concept")
    test_dataloader = du.get_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        split=["test"],
        train_correlation=train_correlation,
        test_correlation=test_correlation,
        batch_size=batch_size,
        seeds=(split_seed, shuffle_seed)
    )
    debiasing_res["chi2_test"] = cu.compute_concept_bias_chi2(test_dataloader, {"merged": craft}, {"merged": concept_parameters["concept_parameters"].keys()}, device)
    with open(debiasing_path, "wb") as f:
        pkl.dump(debiasing_res, f)

#  Phase 4: Debiasing the model by removing the bias concept from the latent space and observe the impact on the performance
if ("debiasing_impact" not in debiasing_res or 4 in args.overwrite_phase) and len(debiasing_res["bias_merged"]) > 0:
    print("Phase 4: Evaluating the debiasing impact")
    test_dataloader = du.get_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        split=["test"],
        train_correlation=train_correlation,
        test_correlation=test_correlation,
        batch_size=batch_size,
        seeds=(split_seed, shuffle_seed)
    )
    bias_list = debiasing_res["bias_merged"]
    if len(bias_list) > 1:
        bias_list = bias_list[:1]
        print(f"Too much bias concepts to remove at once, keeping only the first 4 : {bias_list}")
    debiasing_res["debiasing_impact"] = cu.evaluate_debiasing_impact(test_dataloader, g, h, craft, bias_list, device)
    with open(debiasing_path, "wb") as f:
        pkl.dump(debiasing_res, f)

# Phase 5: Ablation study
if ("ablation_results" not in debiasing_res or 5 in args.overwrite_phase) and len(debiasing_res["bias_merged"]) > 0:
    print("Phase 5: Ablation study on the random concepts")
    test_dataloader = du.get_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        split=["test"],
        train_correlation=train_correlation,
        test_correlation=test_correlation,
        batch_size=batch_size,
        seeds=(split_seed, shuffle_seed)
    )
    debiasing_res["ablation_results"] = []
    for i in range(1):
        print(f"Running ablation study iteration {i+1}/10")
        bias_labels = random.sample(list(range(craft.number_of_concepts)), k=len(debiasing_res["bias_merged"]))
        debiasing_res["ablation_results"].append((bias_labels, cu.evaluate_debiasing_impact(test_dataloader, g, h, craft, bias_labels, device)))
    with open(debiasing_path, "wb") as f:
        pkl.dump(debiasing_res, f)


# Phase 6: Bias amplification study
if ("bias_amplification" not in debiasing_res or 6 in args.overwrite_phase) and len(debiasing_res["bias_merged"]) > 0:
    print("Phase 6: Evaluating the bias amplification")
    test_dataloader = du.get_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        split=["test"],
        train_correlation=train_correlation,
        test_correlation=test_correlation,
        batch_size=batch_size,
        seeds=(split_seed, shuffle_seed)
    )
    debiasing_res["bias_amplification"] = cu.evaluate_bias_amplification(test_dataloader, g, h, craft, debiasing_res["bias_merged"][0:1], device)

with open(debiasing_path, "wb") as f:
    pkl.dump(debiasing_res, f)

# run.finish()
print(f"OOOOOOOOO Debiasing analysis experiment number {model_id}|{concept_id} runned with sucess OOOOOOOOO")

