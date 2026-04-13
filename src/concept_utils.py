import torch
import numpy as np
from torch import nn
from craft.craft_torch import torch_to_numpy
from torch.linalg import vector_norm as norm
from scipy.stats import chi2_contingency
from sklearn.metrics import matthews_corrcoef
import pickle as pkl
import src.model_utils as mu

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



def gather_all_concept_results(dataloader, model, loss_fn, concept_engines, gradient_recoverer, backprop_mult=1, device="cpu"):
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
        else:
            concept_engine = list(concept_engines.values())[0]
        results[bias_label] = {}
        for Xunbiased, Xbiased, y in bias_dataloader:
            unbiased_concept_activation, biased_concept_activation, activation_difference = get_bias_concepts(Xunbiased, Xbiased, model, concept_engine, device)
            for label in y.unique():
                label = int(label)
                if label in results[bias_label]:
                    results[bias_label][label] = (
                                    np.concatenate((results[bias_label][label][0], unbiased_concept_activation[y == label])),
                                    np.concatenate((results[bias_label][label][1], biased_concept_activation[y == label])),
                                    np.concatenate((results[bias_label][label][2], activation_difference[y == label].detach()))
                                    )
                else :
                    results[bias_label][label] = (
                        unbiased_concept_activation[y==label],
                        biased_concept_activation[y==label],
                        activation_difference[y==label].detach()
                    )
    return results


def compute_concept_bias_chi2(dataloader, concept_engines, biases, device="cpu"):
    """
    Computes Chi-squared test statistic and p-value between each concept and each bias.
    
    Performs a single loop through the dataloader and extracts latent activations for each
    concept engine. Builds contingency tables on-the-fly without keeping activations in memory.
    For each concept engine, computes Chi-squared test statistics between each concept 
    (binarized as presence/absence) and each corresponding bias.
    
    Args:
        dataloader: DataLoader with batches of (X, y, ...)
        concept_engines: Dict where keys are concept engine names and values are concept 
                        engine instances
        biases: Dict where keys match concept_engines keys, and values are dicts of 
               biases for that concept engine (e.g., biases[key] = {bias_name: bias_values})
        device: Device to run computations on (default: "cpu")
    
    Returns:
        chi2_results: Dict mapping each concept engine key to a dictionary of chi-squared 
                     results for each bias. Format: {engine_key: {bias_name: {
                         'chi2_stats': array, 'p_values': array}}}
                     
    Examples:
        >>> chi2_results = compute_concept_bias_chi2(test_loader, concept_engines, biases, device="cuda")
        >>> print(chi2_results['engine_key']['bias_name']['chi2_stats'])
    """
    chi2_results = {}
    
    # Initialize contingency table counters for all engine/bias/concept combinations
    # Structure: {engine_key: {bias_name: {concept_idx: contingency_table}}}
    contingency_tables = {}
    n_concepts_per_engine = {}
    
    # First pass: get number of concepts for each engine (process first batch)
    with torch.no_grad():
        batch = next(iter(dataloader))
        X = batch[0]
        for engine_key, concept_engine in concept_engines.items():
            activation = concept_engine.input_to_latent(X.to(device))
            concept_activation = concept_engine.transform(inputs=None, activations=activation)
            n_concepts = concept_activation.shape[1]
            n_concepts_per_engine[engine_key] = n_concepts
    
    # Initialize contingency tables
    for engine_key in concept_engines.keys():
        contingency_tables[engine_key] = {}
        engine_biases = biases[engine_key]
        for bias_name in engine_biases:
            contingency_tables[engine_key][bias_name] = {}
            for concept_idx in range(n_concepts_per_engine[engine_key]):
                contingency_tables[engine_key][bias_name][concept_idx] = np.zeros((2, 2))
    
    # Single pass through dataloader, building contingency tables
    sample_offset = 0  # Track position in the flattened bias arrays
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X = batch[0]
            batch_size = X.shape[0]
            
            # Process each concept engine
            for engine_key, concept_engine in concept_engines.items():
                # Get latent activations using this concept engine
                activation = concept_engine.input_to_latent(X.to(device))
                
                # Transform to concepts
                concept_activation = concept_engine.transform(inputs=None, activations=activation)
                
                # Binarize concepts: 1 if activation > 0, else 0
                concepts_binary = (concept_activation > 0).astype(int)  # (batch_size, n_concepts)
                
                # Get biases for this engine
                engine_biases = biases[engine_key]
                
                # Update contingency tables for each bias
                for bias_value in engine_biases:
                    bias_present = batch[2] == bias_value  # Assuming bias labels are in order after X and y
                    batch_bias_values = bias_present.cpu().numpy() if torch.is_tensor(bias_present) else bias_present
                    # Update contingency table for each concept
                    for concept_idx in range(concepts_binary.shape[1]):
                        contingency_tables[engine_key][bias_value][concept_idx] += np.array([[((batch_bias_values == 0) & (concepts_binary[:, concept_idx] == 0)).sum(),
                                ((batch_bias_values == 0) & (concepts_binary[:, concept_idx] == 1)).sum()],
                            [((batch_bias_values == 1) & (concepts_binary[:, concept_idx] == 0)).sum(),
                                ((batch_bias_values == 1) & (concepts_binary[:, concept_idx] == 1)).sum()]])
    
    # Compute chi-squared statistics from contingency tables
    for engine_key in concept_engines.keys():
        chi2_results[engine_key] = {}
        engine_biases = biases[engine_key]
        
        for bias_name in engine_biases:
            chi2_stats = np.zeros(n_concepts_per_engine[engine_key])
            p_values = np.zeros(n_concepts_per_engine[engine_key])
            mcc_values = np.zeros(n_concepts_per_engine[engine_key])
            
            for concept_idx in range(n_concepts_per_engine[engine_key]):
                contingency_table = contingency_tables[engine_key][bias_name][concept_idx]
                
                # Perform chi-squared test
                try:
                    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
                    chi2_stats[concept_idx] = chi2
                    p_values[concept_idx] = p_val
                    mcc_true = []
                    mcc_pred = []
                    for i, row in enumerate(contingency_table):
                        for j, count in enumerate(row):
                            mcc_true.extend([i] * int(count))
                            mcc_pred.extend([j] * int(count))
                    mcc_values[concept_idx] = matthews_corrcoef(mcc_true, mcc_pred)
                except Exception as e:
                    # Handle cases where chi-squared test fails (e.g., insufficient data)
                    chi2_stats[concept_idx] = np.nan
                    p_values[concept_idx] = np.nan
                    mcc_values[concept_idx] = np.nan

            chi2_results[engine_key][bias_name] = {
                'chi2_stats': chi2_stats,
                'p_values': p_values,
                'mcc_values': mcc_values
            }
    
    return chi2_results




def get_bias_estimator(data_folder, exp_name, exp_id, exp_type, concept_id, patch_id, backprop_step, bias_threshold, mode="paper"):
    number_of_concept = concept_id
    with open(f"{data_folder}models/{exp_name}/concepts_{exp_id}_{exp_type}_{concept_id}_{patch_id}.pkl", "rb") as f:
        concept_parameters = pkl.load(f)
    concept_res = concept_parameters["concept_results"][backprop_step]
    debias_results = {"bias_vectors": [], "bias_threshold": bias_threshold, "all_values": []}

    for label in concept_res.keys():
        if mode == "ideal":
            if "false_n" in concept_res.get(label, {}):
                results_fn = concept_res[label]["false_n"]
                valid_n = results_fn["concept_base"] <= 0
                added_n = ((results_fn["concept_descent"] > 0) & valid_n).sum(axis=0)
                valid_n = valid_n.sum(axis=0)
            else:
                valid_n = 0

            if "false_p" in concept_res.get(label, {}):
                results_fp = concept_res[label]["false_p"]
                results_fp = concept_res[label]["false_p"]
                valid_p = results_fp["concept_base"] > 0
                removed_p = ((results_fp["concept_descent"] <= 0) & valid_p).sum(axis=0)
                valid_p = valid_p.sum(axis=0)
            else:
                valid_p = 0
                removed_p = 0
            
            general_valid = valid_n + valid_p
            general_valid = general_valid + general_valid.sum()/(3*number_of_concept)
            final_res = np.zeros(number_of_concept)

            if isinstance(general_valid, int):
                final_res = 0
            else:
                final_res[general_valid > 0] = (added_n + removed_p)[general_valid > 0] / general_valid[general_valid > 0]
        

        # final_res = np.array([max(false_p_res[k], false_n_res[k]) for k in range(len(false_p_res))])
        # f = lambda x, y: 0 if (x < 0 and y < 0) else (x if y < 0 else (y if x < 0 else (x + y) / 2))
        # final_res = np.array([f(false_p_res[k], false_n_res[k]) for k in range(len(false_p_res))])
        if mode == "paper":
            if "false_n" in concept_res[label]:
                results = concept_res[label]["false_n"]
                appearance = len(results["concept_base"])
                base_freq_n = ((results["concept_base"] > 0).sum(axis=0))/appearance
                descent_freq_n = (results["concept_descent"] > 0).sum(axis=0)/appearance
            else:
                base_freq_n = np.array([0 for i in range(number_of_concept)])
                descent_freq_n = np.array([0 for i in range(number_of_concept)])
            false_n_res = descent_freq_n - base_freq_n
            false_n_res[false_n_res < 0] = 0
            if "false_p" in concept_res[label]:
                results = concept_res[label]["false_p"]
                appearance = len(results["concept_base"])
                base_freq_p = ((results["concept_base"] > 0).sum(axis=0))/appearance
                descent_freq_p = (results["concept_descent"] > 0).sum(axis=0)/appearance
            else: 
                base_freq_p = np.array([0 for i in range(number_of_concept)])
                descent_freq_p = np.array([0 for i in range(number_of_concept)])
            false_p_res = base_freq_p - descent_freq_p
            false_p_res[false_p_res < 0] = 0

            final_res = (false_p_res + false_n_res) /2
        # final_res = (base_freq_p + descent_freq_n) - (base_freq_n + descent_freq_p)

        debias_results[label] = final_res
        debias_results["all_values"].append(final_res)
        for rank, value in enumerate(final_res):
            if value > bias_threshold:
                debias_results["bias_vectors"].append((label, rank))
    return debias_results



class ActivationDisturber():
    def __init__(self,  layer, concepts, craft, device = "cuda"):
        self.craft_decomposer = craft
        self.bias_concepts = concepts
        self.handle = layer.register_forward_hook(self._hook_function)
        self.device = device
        print(f"Bias concepts: {self.bias_concepts}")
    
    def _hook_function(self, model, input, output):
        # print(self.bias_concepts)
        concepts_activation = self.craft_decomposer.transform(
            inputs=None, activations=output
        )
        correction = torch.Tensor(
            concepts_activation[:, self.bias_concepts]
            @ self.craft_decomposer.W[self.bias_concepts]
        ).to(self.device)
        mult = norm(output, ord=2, dim=1) / (norm(output - correction, ord=2, dim=1) + 1e-30)
        return (output - correction) * mult.unsqueeze(1)

    def __del__(self):
        self.handle.remove()

def evaluate_debiasing_impact(dataloader, g, h, concept_decomposer, bias_labels, device="cpu"):
    ad = ActivationDisturber(layer = g[-1], concepts=bias_labels, craft=concept_decomposer, device=device)
    n_model = nn.Sequential(g,h)
    res = mu.test(dataloader, n_model, nn.CrossEntropyLoss(), device=device)
    ad.handle.remove()
    del(ad)
    return res


class ActivationBiaser():
    def __init__(self, layer, concepts, craft, device="cuda"):
        self.craft_decomposer = craft
        self.bias_concepts = concepts
        self.handle = layer.register_forward_hook(self._hook_function)
        self.device = device
    
    def _hook_function(self, model, input, output):
        concepts_activation = self.craft_decomposer.transform(
            inputs=None, activations=output
        )
        
        # Extract bias concepts
        bias_concept_activation = concepts_activation[:, self.bias_concepts]
        
        # For each sample, compute average activation of activated bias concepts (>0)
        activated_mask = bias_concept_activation > 0
        sum_activated = (bias_concept_activation * activated_mask).sum(axis=1)
        count_activated = activated_mask.sum(axis=1)
        
        # Average activation per sample - avoid division by zero
        avg_activated = np.divide(
            sum_activated,
            count_activated,
            where=count_activated != 0,
            out=np.zeros_like(sum_activated)
        )
        
        # Amplify bias concepts: add the average activation to each concept
        amplified_bias_concepts = avg_activated[:, np.newaxis] * 20
        
        amplified_contribution = torch.Tensor(
            amplified_bias_concepts @ self.craft_decomposer.W[self.bias_concepts]
        ).to(self.device)
        
        # Modify output by adding the amplification difference
        modified_output = output + amplified_contribution
        
        # Normalize to maintain L2 norm
        mult = norm(output, ord=2, dim=1) / (norm(modified_output, ord=2, dim=1) + 1e-30)
        return modified_output * mult.unsqueeze(1)

    def __del__(self):
        self.handle.remove()


def evaluate_bias_amplification(dataloader, g, h, concept_decomposer, bias_labels, device="cpu"):
    n_model = nn.Sequential(g,h)
    res_base = mu.adjacency_test(dataloader, n_model, nn.CrossEntropyLoss(), device=device)
    ad = ActivationBiaser(layer = g[-1], concepts=bias_labels, craft=concept_decomposer, device=device)
    n_model = nn.Sequential(g,h)
    res_biased = mu.adjacency_test(dataloader, n_model, nn.CrossEntropyLoss(), device=device)
    bias_res_biased = mu.test(dataloader, n_model, nn.CrossEntropyLoss(), device=device)
    ad.handle.remove()
    del(ad)
    return res_base, res_biased