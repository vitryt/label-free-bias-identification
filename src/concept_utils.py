import torch
import numpy as np
from torch import nn
from craft.craft_torch import torch_to_numpy

class Gradient_retriever():
    def __init__(self, layer):
        self.grad = None
        self.handle = layer.register_full_backward_hook(self._hook_function)
    
    def _hook_function(self, m, input_g, output_g):
        self.grad = torch.clone(input_g[0])
    
    def __del__(self):
        self.handle.remove()

def compute_mask_patches(inputs, patch_size):
    strides = int(patch_size * 0.80)
    patches = torch.nn.functional.unfold(inputs, kernel_size=patch_size, stride=strides)
    patches = patches.transpose(1, 2).contiguous().view(-1, 1, patch_size, patch_size)
    return patches

def train_concept_engines(dataloader, model, layer_depth, number_of_concept, concept_decomposer, patch_size, concept_dataset_size=3000, batch_size=256, device="cpu", get_masks = False):
    concept_engines = {}
    concept_parameters = {}
    concept_datasets = {}
    if get_masks:
        concept_masks = {}
    model.eval()
    model.zero_grad()
    for batch in dataloader:
        X = batch[0]
        y = model(X.to(device)).cpu().argmax(1)
        if get_masks:
            masks = batch[-1]
        for label in y.unique():
            label = int(label)
            if not label in concept_datasets:
                print(f"------------------------------------Added label {label}")
                concept_datasets[label] = X[y==label]
                if get_masks: concept_masks[label] = masks[y==label]
            elif len(concept_datasets[label]) < concept_dataset_size:
                concept_datasets[label] = torch.cat([concept_datasets[label], X[y==label]])
                if get_masks: concept_masks[label] = torch.cat([concept_masks[label], masks[y==label]])

    if hasattr(model, "backbone"):
         backbone = model.backbone
         g = nn.Sequential(*(list(model.backbone.children())[:-1] + [nn.Flatten(start_dim=1)])) # Layers pre concept decomposition
         h = nn.Sequential(backbone.fc) # Layers post concept decompositon
    else:
        g = nn.Sequential(*(list(model.children())[:layer_depth-1])) # Layers pre concept decomposition (CMNIST version)
        h = nn.Sequential(*(list(model.children())[layer_depth-1:])) # Layers post concept decompositon (CMNIST version)

    for label, dataset in concept_datasets.items():
        print(f"Training concept decomposer for class {label}")
        if len(dataset > concept_dataset_size):
            dataset = dataset[:concept_dataset_size]
        concept_engines[label] = concept_decomposer(
            input_to_latent=g,
            latent_to_logit = h,
            number_of_concepts = number_of_concept,
            patch_size = patch_size,
            batch_size = batch_size,
            device = device
        )
        concept_parameters[label] = {}
        concept_parameters[label]["crops"], concept_parameters[label]["crops_u"], concept_parameters[label]["W"] = concept_engines[label].fit(concept_datasets[label])
        concept_parameters[label]["reducer"] = concept_engines[label].reducer
        concept_parameters[label]["crops"] = np.moveaxis(torch_to_numpy(concept_parameters[label]["crops"]), 1, -1)
        if get_masks:
            concept_parameters[label]["crop_masks"] = compute_mask_patches(concept_masks[label], patch_size)
            concept_parameters[label]["crop_masks"] = np.moveaxis(torch_to_numpy(concept_parameters[label]["crop_masks"]), 1, -1)
    return concept_engines, concept_parameters


def get_backprop(X, y, model, loss_fn, concept_engine, gradient_recoverer, back_mult = 1, device="cpu", activation = None):
    """
    Returns :
    - concept_activation : the activations for each input X
    - neg_concept_diff : the concept activation in the direction of the backpropagated gradient descent
    - pos_concept_diff : the concept activation in the direction of the backpropagated gradient ascent
    """
    if activation == None:
        activation = concept_engine.input_to_latent(X.cuda()).cpu()
    model.eval()
    model.zero_grad()
    outpt = concept_engine.latent_to_logit(activation.to(device)).cpu()
    loss = loss_fn(outpt, y)
    loss.backward()
    return gradient_recoverer.grad * back_mult



def gather_all_concept_results(dataloader, model, loss_fn, concept_engines, gradient_recoverer, backprop_mult=1, device="cpu", multi_concept=False):
    results = {}
    concept_engine = list(concept_engines.values())[0]
    output_size = 0
    activation_wrong = torch.ByteTensor().to(device)
    y_wrong = torch.LongTensor()
    y_predi_wrong = torch.ByteTensor()
    backprop_wrong = torch.ByteTensor().to(device)
    for i, batch in enumerate(dataloader):
        # print(f"Doing batch {i}                            ", end="\r")
        X = batch[0]
        y = batch[1]

        activation = concept_engine.input_to_latent(X.to(device))
        y_predi = (concept_engine.latent_to_logit(activation).cpu())
        if output_size == 0:
            output_size = len(y_predi[0])
            results = {}
        y_predi = y_predi.argmax(dim=1)
        wrongly_classified = (y_predi != y)
        activation_wrong = torch.cat([activation_wrong, activation[wrongly_classified]])
        y_wrong = torch.cat([y_wrong, y[wrongly_classified]])
        y_predi_wrong = torch.cat([y_predi_wrong, y_predi[wrongly_classified]])
        backprop_wrong = torch.cat(
            [
                backprop_wrong,
                get_backprop(
                    X = None,
                    y = y[wrongly_classified],
                    model = model,
                    loss_fn = loss_fn,
                    concept_engine = concept_engine,
                    gradient_recoverer = gradient_recoverer,
                    back_mult=backprop_mult,
                    device=device,
                    activation=activation[wrongly_classified]
                    )
            ]
        )
    for label in y_wrong.unique():
        label = int(label)
        m = concept_engine.number_of_concepts
        studied_index = np.array(range(len(y_wrong)))[y_wrong==label]
        # n = len(studied_index)
        # k = n
        # batch_activation = []
        # batch_backprop = []
        # for j in range(k):
        #     print(f"Generating datapoint {j}/{k}   ", end="\r")
        #     batch_indices = np.random.choice(studied_index, m)
        #     batch_activation.append(activation_wrong[batch_indices].mean(axis=0))
        #     batch_backprop.append(backprop_wrong[batch_indices].mean(axis=0))
        # batch_activation = torch.stack(batch_activation)
        # batch_backprop = torch.stack(batch_backprop)
        batch_activation = activation_wrong[studied_index]
        batch_backprop = backprop_wrong[studied_index]
        
        batch_size = concept_engines[label].batch_size
        batch_amount = len(batch_activation)/batch_size
        batch_amount = int(batch_amount) + (1 if batch_amount != int(batch_amount) else 0)
        batch_base = []
        batch_descents = []
        batch_ascents = []
        for batch_id in range(batch_amount):
            print(f"Batch {batch_id}/{batch_amount}           ", end="\r")
            if batch_id < batch_amount -1:
                batch_indices = range(batch_id * batch_size, (batch_id+1) * batch_size)
            else:
                batch_indices = range(batch_id * batch_size, len(batch_activation))
            
            concept_activation = concept_engines[label].transform(inputs=None, activations=batch_activation[batch_indices])
            batch_base.append(concept_activation +1 -1)
            modified_activation = batch_activation[batch_indices] - batch_backprop[batch_indices]
            modified_activation[(modified_activation) < 0] = 0
            back_concept_activation = concept_engines[label].transform(inputs=None, activations = modified_activation)
            batch_descents.append(back_concept_activation +1 -1)

            modified_activation = batch_activation[batch_indices] + batch_backprop[batch_indices]
            modified_activation[(modified_activation) < 0] = 0
            back_concept_activation = concept_engines[label].transform(inputs=None, activations = modified_activation)
            batch_ascents.append(back_concept_activation +1 -1)
        concept_base, concept_ascent, concept_descent = np.concatenate(batch_base), np.concatenate(batch_ascents), np.concatenate(batch_descents)
        if label not in results: results[label] = {}
        results[label]["false_n"] = {
            "concept_base":concept_base,
            "concept_descent":concept_descent,
            "concept_ascent":concept_ascent,
            "backprop_vector":batch_backprop.cpu(),
        }

    for label in y_predi_wrong.unique():
        label = int(label)
        m = concept_engine.number_of_concepts
        studied_index = np.array(range(len(y_predi_wrong)))[y_predi_wrong==label]
        # n = len(studied_index)
        # k = n
        # batch_activation = []
        # batch_backprop = []
        # for j in range(k):
        #     print(f"Generating datapoint {j}/{k}   ", end="\r")
        #     batch_indices = np.random.choice(studied_index, m)
        #     batch_activation.append(activation_wrong[batch_indices].mean(axis=0))
        #     batch_backprop.append(backprop_wrong[batch_indices].mean(axis=0))
        # batch_activation = torch.stack(batch_activation)
        # batch_backprop = torch.stack(batch_backprop)
        batch_activation = activation_wrong[studied_index]
        batch_backprop = backprop_wrong[studied_index]
        
        batch_size = concept_engines[label].batch_size
        batch_amount = len(batch_activation)/batch_size
        batch_amount = int(batch_amount) + (1 if batch_amount != int(batch_amount) else 0)
        batch_base = []
        batch_descents = []
        batch_ascents = []
        for batch_id in range(batch_amount):
            print(f"Batch {batch_id}/{batch_amount}           ", end="\r")
            if batch_id < batch_amount -1:
                batch_indices = range(batch_id * batch_size, (batch_id+1) * batch_size)
            else:
                batch_indices = range(batch_id * batch_size, len(batch_activation))
            
            concept_activation = concept_engines[label].transform(inputs=None, activations=batch_activation[batch_indices])
            batch_base.append(concept_activation +1 -1)
            modified_activation = batch_activation[batch_indices] - batch_backprop[batch_indices]
            modified_activation[(modified_activation) < 0] = 0
            back_concept_activation = concept_engines[label].transform(inputs=None, activations = modified_activation)
            batch_descents.append(back_concept_activation +1 -1)

            modified_activation = batch_activation[batch_indices] + batch_backprop[batch_indices]
            modified_activation[(modified_activation) < 0] = 0
            back_concept_activation = concept_engines[label].transform(inputs=None, activations = modified_activation)
            batch_ascents.append(back_concept_activation +1 -1)
        concept_base, concept_ascent, concept_descent = np.concatenate(batch_base), np.concatenate(batch_ascents), np.concatenate(batch_descents)
        if label not in results: results[label] = {}
        results[label]["false_p"] = {
            "concept_base":concept_base,
            "concept_descent":concept_descent,
            "concept_ascent":concept_ascent,
            "backprop_vector":batch_backprop.cpu(),
        }
    return results


def analyze_results(results, i, number_of_concept):
    # Count concepts that were added/removed when doing gradient descent
    appearance = len(results[i]["concept_base"])
    d_added_concept = ((results[i]["concept_base"] == 0) & (results[i]["concept_descent"] > 0)).sum(axis=0)
    d_removed_concept = ((results[i]["concept_base"] > 0) & (results[i]["concept_descent"] == 0)).sum(axis=0)
    # Count concepts that were added/removed when doing gradient ascend
    a_added_concept = ((results[i]["concept_base"] == 0) & (results[i]["concept_ascent"] > 0)).sum(axis=0)
    a_removed_concept = ((results[i]["concept_base"] > 0) & (results[i]["concept_ascent"] == 0)).sum(axis=0)
    raw_d_diff = results[i]["concept_descent"].sum(axis=0) - results[i]["concept_base"].sum(axis=0)
    raw_a_diff = results[i]["concept_ascent"].sum(axis=0) - results[i]["concept_base"].sum(axis=0)

    return {
        "data_size":appearance,
        "descent_added_concept":d_added_concept,
        "descent_removed_concept":d_removed_concept,
        "ascent_added_concept":a_added_concept,
        "ascent_removed_concept":a_removed_concept,
        "descent_raw_difference":raw_d_diff,
        "ascent_raw_difference":raw_a_diff,
    }
def get_bias_concepts(Xunbiased, Xbiased, model, concept_engine, device="cpu"):
    """
    """
    unbiased_activation = concept_engine.input_to_latent(Xunbiased.to(device)).cpu()
    unbiased_concept_activation = concept_engine.transform(inputs=None, activations=unbiased_activation)

    biased_activation = concept_engine.input_to_latent(Xbiased.to(device)).cpu()
    biased_concept_activation = concept_engine.transform(inputs=None, activations=biased_activation)

    return unbiased_concept_activation, biased_concept_activation, biased_activation - unbiased_activation

def gather_all_bias_results(bias_dataloaders, model, concept_engines, device="cpu"):
    concept_engine = list(concept_engines.values())[0]
    results = {}
    output_size = 0
    for bias_label, bias_dataloader in bias_dataloaders.items():
        if len(concept_engines)>1:
            if bias_label in concept_engines:
                concept_engine = concept_engines[bias_label]
            else:
                raise KeyError(f"the bias labeled {bias_label} has no attributed concept decomposer device")
        results[bias_label] = {}
        for Xunbiased, Xbiased, y in bias_dataloader:
            # y_predi = (model(X.cuda()).cpu())
            # if output_size == 0:
            #     output_size = len(y_predi[0])
            #     results = {}
            # y_predi = y_predi.argmax(dim=1)
            # wrongly_classified = (y_predi != y)
            # X_wrong = X[wrongly_classified]
            # y_wrong = y[wrongly_classified]
            # y_predi_wrong = y_predi[wrongly_classified]
            unbiased_concept_activation, biased_concept_activation, activation_difference = get_bias_concepts(Xunbiased, Xbiased, model, concept_engine, device)
            for label in y.unique():
                label = int(label)
                if label in results[bias_label]:
                    results[bias_label][label] = (
                                    np.concat((results[bias_label][label][0], unbiased_concept_activation[y == label])),
                                    np.concat((results[bias_label][label][1], biased_concept_activation[y == label])),
                                    np.concat((results[bias_label][label][2], activation_difference[y == label].detach()))
                                    )
                else :
                    results[bias_label][label] = (
                        unbiased_concept_activation[y==label],
                        biased_concept_activation[y==label],
                        activation_difference[y==label].detach()
                    )
    return results