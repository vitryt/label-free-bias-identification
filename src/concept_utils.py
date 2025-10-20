import torch
import numpy as np

class Gradient_retriever():
    def __init__(self, layer):
        self.grad = None
        self.handle = layer.register_full_backward_hook(self._hook_function)
    
    def _hook_function(self, m, input_g, output_g):
        self.grad = torch.clone(input_g[0])
    
    def __del__(self):
        self.handle.remove()


def get_backprop_concepts(X, y, model, loss_fn, concept_engine, gradient_recoverer, back_mult = 1, device="cpu", activation = None):
    """
    Returns :
    - concept_activation : the activations for each input X
    - neg_concept_diff : the concept activation in the direction of the backpropagated gradient descent
    - pos_concept_diff : the concept activation in the direction of the backpropagated gradient ascent
    """
    if activation == None:
        activation = concept_engine.input_to_latent(X.cuda()).cpu()
    concept_activation = concept_engine.transform(inputs=None, activations=activation)
    model.eval()
    model.zero_grad()
    outpt = concept_engine.latent_to_logit(activation.to(device)).cpu()
    loss = loss_fn(outpt, y)
    loss.backward()
    back_activation = gradient_recoverer.grad.cpu()
    modified_activation = activation - (back_activation * back_mult)
    modified_activation[(modified_activation) < 0] = 0
    back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
    neg_concept_diff = back_concept_activation +1 -1
    # neg_concept_diff = (back_concept_activation - concept_activation)
    modified_activation = activation + (back_activation * back_mult)
    modified_activation[(modified_activation) < 0] = 0
    back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
    pos_concept_diff = back_concept_activation +1 -1
    # pos_concept_diff = (back_concept_activation - concept_activation)
    return concept_activation, neg_concept_diff, pos_concept_diff


def gather_all_concept_results(dataloader, model, loss_fn, concept_engine, gradient_recoverer, backprop_mult=1, device="cpu"):
    results = {}
    output_size = 0
    for X, y, _ in dataloader:
        y_predi = (model(X.cuda()).cpu())
        if output_size == 0:
            output_size = len(y_predi[0])
            results = {}
        y_predi = y_predi.argmax(dim=1)
        wrongly_classified = (y_predi != y)
        X_wrong = X[wrongly_classified]
        y_wrong = y[wrongly_classified]
        y_predi_wrong = y_predi[wrongly_classified]
        if len(y_wrong > 0):
            concept_activation, neg_concept_diff, pos_concept_diff = get_backprop_concepts(
                X = X_wrong,
                y = y_wrong,
                model = model,
                loss_fn = loss_fn,
                concept_engine = concept_engine,
                gradient_recoverer = gradient_recoverer,
                back_mult = backprop_mult,
                device = device,
            )
            for i in range(output_size):
                if not i in results:
                    results[i] = {}
                for j in range(output_size):
                    if int(((y_wrong == i) & (y_predi_wrong == j)).sum()) > 0:
                        indice = [0] if len(concept_activation) == 1 else (y_wrong == i) & (y_predi_wrong == j)
                        if not j in results[i]:
                            results[i][j] = (
                                concept_activation[indice],
                                pos_concept_diff[indice],
                                neg_concept_diff[indice],
                            )
                        else:
                            results[i][j] = (
                                np.concat((results[i][j][0], concept_activation[indice])),
                                np.concat((results[i][j][1], neg_concept_diff[indice])),
                                np.concat((results[i][j][2], pos_concept_diff[indice]))
                                )
    return results


def analyze_results(results, i, number_of_concept):
    # Count concepts that were added/removed when doing gradient descent
    n_appearance = np.ones(number_of_concept)
    n_added_bias = np.zeros(number_of_concept)
    n_removed_bias = np.zeros(number_of_concept)
    # Count concepts that were added/removed when doing gradient ascend
    p_appearance = np.ones(number_of_concept)
    p_added_bias = np.zeros(number_of_concept)
    p_removed_bias = np.zeros(number_of_concept)
    raw_n_diff = np.zeros(number_of_concept)
    raw_p_diff = np.zeros(number_of_concept)
    for key1, val1 in results.items():
        for key2, val2 in val1.items():
            if key1 == i:
                n_appearance += len(val2[0])
                # n_appearance += ((val2[0] == 0)).sum(axis=0)
                n_added_bias += ((val2[0]==0) & (val2[1]>0)).sum(axis=0)
                n_removed_bias += ((val2[0]>0) & (val2[1]==0)).sum(axis=0)
                raw_n_diff += (val2[1] - val2[0]).sum(axis=0)
            if key2 == i:
                p_appearance += len(val2[0])
                # p_appearance += (val2[0] > 0).sum(axis=0)
                p_added_bias += ((val2[0]==0) & (val2[2]>0)).sum(axis=0)
                p_removed_bias += ((val2[0]>0) & (val2[2]==0)).sum(axis=0)
                raw_p_diff += (val2[2] - val2[0]).sum(axis=0)
    return n_added_bias, n_removed_bias, n_appearance, p_added_bias, p_removed_bias, p_appearance, raw_n_diff/n_appearance, raw_p_diff/p_appearance


def get_bias_concept(results, studied_class, number_of_concepts):
    n_added_bias, n_removed_bias, n_appearance, p_removed_bias, p_appearance = analyze_results(results, studied_class, number_of_concept=number_of_concepts)
    na_biases = {}
    for i, val in enumerate(n_added_bias/n_appearance):
        na_biases[i]= val
    nr_biases = {}
    for i, val in enumerate(n_removed_bias):
        nr_biases[i]= val
    pr_biases = {}
    for i, val in enumerate(p_removed_bias/p_appearance):
        pr_biases[i]= val
    ultimate_bias = {}
    for key, val in na_biases.items():
        ultimate_bias[key]= val - pr_biases[key]
    return ultimate_bias



def get_bias_concepts(Xunbiased, Xbiased, model, concept_engine, device="cpu"):
    """
    """
    unbiased_activation = concept_engine.input_to_latent(Xunbiased.cuda()).cpu()
    unbiased_concept_activation = concept_engine.transform(inputs=None, activations=unbiased_activation)

    biased_activation = concept_engine.input_to_latent(Xbiased.cuda()).cpu()
    biased_concept_activation = concept_engine.transform(inputs=None, activations=biased_activation)

    return unbiased_concept_activation, biased_concept_activation

def gather_all_bias_results(bias_dataloaders, model, concept_engine, device="cpu"):
    results = {}
    output_size = 0
    for bias_label, bias_dataloader in bias_dataloaders.items():
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




def get_backprop_activations(X, y, model, loss_fn, concept_engine, gradient_recoverer, back_mult = 1, device="cpu"):
    """
    Returns :
    - concept_activation : the activations for each input X
    - neg_concept_diff : the concept activation in the direction of the backpropagated gradient descent
    - pos_concept_diff : the concept activation in the direction of the backpropagated gradient ascent
    """
    activation = concept_engine.input_to_latent(X.cuda()).cpu()
    concept_activation = concept_engine.transform(inputs=None, activations=activation)
    model.eval()
    model.zero_grad()
    outpt = model(X.to(device)).cpu()
    loss = loss_fn(outpt, y)
    loss.backward()
    back_activation = gradient_recoverer.grad.cpu()
    modified_activation = activation - (back_activation * back_mult)
    modified_activation[(modified_activation) < 0] = 0
    back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
    neg_concept_diff = back_concept_activation +1 -1
    # neg_concept_diff = (back_concept_activation - concept_activation)
    modified_activation = activation + (back_activation * back_mult)
    modified_activation[(modified_activation) < 0] = 0
    back_concept_activation = concept_engine.transform(inputs=None, activations = modified_activation)
    pos_concept_diff = back_concept_activation +1 -1
    # pos_concept_diff = (back_concept_activation - concept_activation)
    return concept_activation, neg_concept_diff, pos_concept_diff


def gather_all_concept_results(dataloader, model, loss_fn, concept_engine, gradient_recoverer, backprop_mult=1, device="cpu"):
    results = {}
    output_size = 0
    for X, y, _ in dataloader:
        y_predi = (model(X.cuda()).cpu())
        if output_size == 0:
            output_size = len(y_predi[0])
            results = {}
        y_predi = y_predi.argmax(dim=1)
        wrongly_classified = (y_predi != y)
        X_wrong = X[wrongly_classified]
        y_wrong = y[wrongly_classified]
        y_predi_wrong = y_predi[wrongly_classified]
        if len(y_wrong > 0):
            concept_activation, neg_concept_diff, pos_concept_diff = get_backprop_concepts(
                X = X_wrong,
                y = y_wrong,
                model = model,
                loss_fn = loss_fn,
                concept_engine = concept_engine,
                gradient_recoverer = gradient_recoverer,
                back_mult = backprop_mult,
                device = device,
            )
            for i in range(output_size):
                if not i in results:
                    results[i] = {}
                for j in range(output_size):
                    if int(((y_wrong == i) & (y_predi_wrong == j)).sum()) > 0:
                        indice = [0] if len(concept_activation) == 1 else (y_wrong == i) & (y_predi_wrong == j)
                        if not j in results[i]:
                            results[i][j] = (
                                concept_activation[indice],
                                pos_concept_diff[indice],
                                neg_concept_diff[indice],
                            )
                        else:
                            results[i][j] = (
                                np.concat((results[i][j][0], concept_activation[indice])),
                                np.concat((results[i][j][1], neg_concept_diff[indice])),
                                np.concat((results[i][j][2], pos_concept_diff[indice]))
                                )
    return results