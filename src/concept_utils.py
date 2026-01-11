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


def train_concept_engines(dataloader, model, layer_depth, number_of_concept, concept_decomposer, patch_size, concept_dataset_size=3000, batch_size=256, device="cpu"):
    concept_engines = {}
    concept_parameters = {}
    concept_datasets = {}
    model.eval()
    model.zero_grad()
    for batch in dataloader:
        X = batch[0]
        y = model(X.to(device)).cpu().argmax(1)
        for label in y.unique():
            label = int(label)
            if not label in concept_datasets:
                print(f"------------------------------------Added label {label}")
                concept_datasets[label] = X[y==label]
            elif len(concept_datasets[label]) < concept_dataset_size:
                concept_datasets[label] = torch.cat([concept_datasets[label], X[y==label]])

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


# def get_backprop_concepts(X, y, model, loss_fn, concept_engine, gradient_recoverer, back_mult = 1, device="cpu", activation = None):
#     """
#     Returns :
#     - concept_activation : the activations for each input X
#     - neg_concept_diff : the concept activation in the direction of the backpropagated gradient descent
#     - pos_concept_diff : the concept activation in the direction of the backpropagated gradient ascent
#     """
#     if activation == None:
#         activation = concept_engine.input_to_latent(X.cuda()).cpu()
#     concept_activation = concept_engine.transform(inputs=None, activations=activation)
#     # model.eval()
#     # model.zero_grad()
#     # outpt = concept_engine.latent_to_logit(activation.to(device)).cpu()
#     # loss = loss_fn(outpt, y)
#     # loss.backward()
#     back_activation = get_backprop(
#         X = None,
#         y = y,
#         model = model,
#         loss_fn = loss_fn,
#         concept_engine = concept_engine,
#         gradient_recoverer = gradient_recoverer,
#         back_mult=back_mult,
#         device=device,
#         activation=activation
#         )
#     modified_activation = activation - back_activation
#     modified_activation[(modified_activation) < 0] = 0
#     back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
#     neg_concept_diff = back_concept_activation +1 -1
#     # neg_concept_diff = (back_concept_activation - concept_activation)
#     modified_activation = activation + back_activation
#     modified_activation[(modified_activation) < 0] = 0
#     back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
#     pos_concept_diff = back_concept_activation +1 -1
#     # pos_concept_diff = (back_concept_activation - concept_activation)
#     return concept_activation, neg_concept_diff, pos_concept_diff


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
    if not multi_concept:
        if len(y_wrong > 0):
            m = concept_engine.number_of_concepts
            for i in range(output_size):
                studied_index = np.array(range(len(y_wrong)))[y_wrong==i]
                n = len(studied_index)
                k = n
                batch_activation = []
                batch_backprop = []
                for j in range(k):
                    batch_indices = np.random.choice(studied_index, m)
                    batch_activation.append(activation_wrong[batch_indices].mean(axis=0))
                    batch_backprop.append(backprop_wrong[batch_indices].mean(axis=0))
                batch_activation = torch.stack(batch_activation)
                batch_backprop = torch.stack(batch_backprop)
                concept_activation = concept_engine.transform(inputs=None, activations=batch_activation)
                modified_activation = batch_activation - batch_backprop
                modified_activation[(modified_activation) < 0] = 0
                back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
                concept_descent = back_concept_activation +1 -1

                modified_activation = batch_activation + batch_backprop
                modified_activation[(modified_activation) < 0] = 0
                back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
                concept_ascent = back_concept_activation +1 -1
                results[i] = {
                    "concept_base":concept_activation,
                    "concept_descent":concept_descent,
                    "concept_ascent":concept_ascent,
                    "backprop_vector":batch_backprop.cpu(),
                }
    else:
        for label in y_wrong.unique():
            label = int(label)
            m = concept_engine.number_of_concepts
            studied_index = np.array(range(len(y_wrong)))[y_wrong==label]
            n = len(studied_index)
            k = n
            batch_activation = []
            batch_backprop = []
            for j in range(k):
                print(f"Generating datapoint {j}/{k}   ", end="\r")
                batch_indices = np.random.choice(studied_index, m)
                batch_activation.append(activation_wrong[batch_indices].mean(axis=0))
                batch_backprop.append(backprop_wrong[batch_indices].mean(axis=0))
            batch_activation = torch.stack(batch_activation)
            batch_backprop = torch.stack(batch_backprop)
            
            batch_size = concept_engines[label].batch_size
            batch_amount = len(batch_activation)/batch_size
            batch_amount = int(batch_amount) + (1 if batch_amount != int(batch_amount) else 0)
            batch_descents = []
            batch_ascents = []
            for batch_id in range(batch_amount):
                print(f"Batch {batch_id}/{batch_amount}           ", end="\r")
                if batch_id < batch_amount -1:
                    batch_indices = range(batch_id * batch_size, (batch_id+1) * batch_size)
                else:
                    batch_indices = range(batch_id * batch_size, len(batch_activation))
                
                concept_activation = concept_engines[label].transform(inputs=None, activations=batch_activation[batch_indices])
                modified_activation = batch_activation[batch_indices] - batch_backprop[batch_indices]
                modified_activation[(modified_activation) < 0] = 0
                back_concept_activation = concept_engines[label].transform(inputs=None, activations = modified_activation)
                batch_descents.append(back_concept_activation +1 -1)

                modified_activation = batch_activation[batch_indices] + batch_backprop[batch_indices]
                modified_activation[(modified_activation) < 0] = 0
                back_concept_activation = concept_engines[label].transform(inputs=None, activations = modified_activation)
                batch_ascents.append(back_concept_activation +1 -1)
            concept_ascent, concept_descent = np.concatenate(batch_ascents), np.concatenate(batch_descents)
            results[label] = {
                "concept_base":concept_activation,
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
    # for key1, val1 in results.items():
    #     for key2, val2 in val1.items():
    #         if key1 == i:
    #             n_appearance += len(val2[0])
    #             # n_appearance += ((val2[0] == 0)).sum(axis=0)
    #             n_added_bias += ((val2[0]==0) & (val2[1]>0)).sum(axis=0)
    #             n_removed_bias += ((val2[0]>0) & (val2[1]==0)).sum(axis=0)
    #             raw_n_diff += (val2[1] - val2[0]).sum(axis=0)
    #         if key2 == i:
    #             p_appearance += len(val2[0])
    #             # p_appearance += (val2[0] > 0).sum(axis=0)
    #             p_added_bias += ((val2[0]==0) & (val2[2]>0)).sum(axis=0)
    #             p_removed_bias += ((val2[0]>0) & (val2[2]==0)).sum(axis=0)
    #             raw_p_diff += (val2[2] - val2[0]).sum(axis=0)
    # return n_added_bias, n_removed_bias, n_appearance, p_added_bias, p_removed_bias, p_appearance, raw_n_diff/n_appearance, raw_p_diff/p_appearance


# def get_bias_concept(results, studied_class, number_of_concepts):
#     n_added_bias, n_removed_bias, n_appearance, p_removed_bias, p_appearance = analyze_results(results, studied_class, number_of_concept=number_of_concepts)
#     na_biases = {}
#     for i, val in enumerate(n_added_bias/n_appearance):
#         na_biases[i]= val
#     nr_biases = {}
#     for i, val in enumerate(n_removed_bias):
#         nr_biases[i]= val
#     pr_biases = {}
#     for i, val in enumerate(p_removed_bias/p_appearance):
#         pr_biases[i]= val
#     ultimate_bias = {}
#     for key, val in na_biases.items():
#         ultimate_bias[key]= val - pr_biases[key]
#     return ultimate_bias



def get_bias_concepts(Xunbiased, Xbiased, model, concept_engine, device="cpu"):
    """
    """
    unbiased_activation = concept_engine.input_to_latent(Xunbiased.to(device)).cpu()
    unbiased_concept_activation = concept_engine.transform(inputs=None, activations=unbiased_activation)

    biased_activation = concept_engine.input_to_latent(Xbiased.to(device)).cpu()
    biased_concept_activation = concept_engine.transform(inputs=None, activations=biased_activation)

    return unbiased_concept_activation, biased_concept_activation

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
            unbiased_concept_activation, biased_concept_activation = get_bias_concepts(Xunbiased, Xbiased, model, concept_engine, device)
            for label in y.unique():
                label = int(label)
                if label in results[bias_label]:
                    results[bias_label][label] = (
                                    np.concat((results[bias_label][label][0], unbiased_concept_activation[y == label])),
                                    np.concat((results[bias_label][label][1], biased_concept_activation[y == label]))
                                    )
                else :
                    results[bias_label][label] = (
                        unbiased_concept_activation[y==label],
                        biased_concept_activation[y==label]
                    )
    return results




# def get_backprop_activations(X, y, model, loss_fn, concept_engine, gradient_recoverer, back_mult = 1, device="cpu"):
#     """
#     Returns :
#     - concept_activation : the activations for each input X
#     - neg_concept_diff : the concept activation in the direction of the backpropagated gradient descent
#     - pos_concept_diff : the concept activation in the direction of the backpropagated gradient ascent
#     """
#     activation = concept_engine.input_to_latent(X.cuda()).cpu()
#     concept_activation = concept_engine.transform(inputs=None, activations=activation)
#     model.eval()
#     model.zero_grad()
#     outpt = model(X.to(device)).cpu()
#     loss = loss_fn(outpt, y)
#     loss.backward()
#     back_activation = gradient_recoverer.grad.cpu()
#     modified_activation = activation - (back_activation * back_mult)
#     modified_activation[(modified_activation) < 0] = 0
#     back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
#     neg_concept_diff = back_concept_activation +1 -1
#     # neg_concept_diff = (back_concept_activation - concept_activation)
#     modified_activation = activation + (back_activation * back_mult)
#     modified_activation[(modified_activation) < 0] = 0
#     back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
#     pos_concept_diff = back_concept_activation +1 -1
#     # pos_concept_diff = (back_concept_activation - concept_activation)
#     return concept_activation, neg_concept_diff, pos_concept_diff


# def gather_all_concept_results(dataloader, model, loss_fn, concept_engine, gradient_recoverer, backprop_mult=1, device="cpu"):
#     results = {}
#     output_size = 0
#     for X, y, _ in dataloader:
#         y_predi = (model(X.cuda()).cpu())
#         if output_size == 0:
#             output_size = len(y_predi[0])
#             results = {}
#         y_predi = y_predi.argmax(dim=1)
#         wrongly_classified = (y_predi != y)
#         X_wrong = X[wrongly_classified]
#         y_wrong = y[wrongly_classified]
#         y_predi_wrong = y_predi[wrongly_classified]
#         if len(y_wrong > 0):
#             concept_activation, neg_concept_diff, pos_concept_diff = get_backprop_concepts(
#                 X = X_wrong,
#                 y = y_wrong,
#                 model = model,
#                 loss_fn = loss_fn,
#                 concept_engine = concept_engine,
#                 gradient_recoverer = gradient_recoverer,
#                 back_mult = backprop_mult,
#                 device = device,
#             )
#             for i in range(output_size):
#                 if not i in results:
#                     results[i] = {}
#                 for j in range(output_size):
#                     if int(((y_wrong == i) & (y_predi_wrong == j)).sum()) > 0:
#                         indice = [0] if len(concept_activation) == 1 else (y_wrong == i) & (y_predi_wrong == j)
#                         if not j in results[i]:
#                             results[i][j] = (
#                                 concept_activation[indice],
#                                 pos_concept_diff[indice],
#                                 neg_concept_diff[indice],
#                             )
#                         else:
#                             results[i][j] = (
#                                 np.concat((results[i][j][0], concept_activation[indice])),
#                                 np.concat((results[i][j][1], neg_concept_diff[indice])),
#                                 np.concat((results[i][j][2], pos_concept_diff[indice]))
#                                 )
#     return results
