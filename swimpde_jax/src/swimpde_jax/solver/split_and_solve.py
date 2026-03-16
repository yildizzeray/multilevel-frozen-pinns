import numpy as np 

def split_and_solve(model, x, y, n_splits):
    """
    Split the neuron matrix into rows that correspond to regions in the weight spectrum.
    """
    model[0].fit(x, y)
    weights0_all = model[0].weights
    biases0_all = model[0].biases
    weight_norms = np.linalg.norm(model[0].weights, axis=0, keepdims=False)
    idx_neurons = np.argsort(weight_norms, axis=0) # can be avoided
    idx_split = np.array_split(idx_neurons, indices_or_sections=max(
        1, n_splits))
    weights = []
    biases = []
    # add an 1 dimension to y_current to make it compatible with y_pred.
    y_current = y.reshape(-1, 1)
    for idx_current in idx_split:
        model[0].weights = weights0_all[..., idx_current]
        model[0].biases = biases0_all[..., idx_current]
        mat_neurons = model[0].transform(x)
        model[1].fit(mat_neurons, y_current)
        weights.append(model[1].weights)
        biases.append(model[1].biases)
        y_pred = model[1].transform(mat_neurons)
        y_current = y_current - y_pred
    model[0].weights = weights0_all[..., idx_neurons]
    model[0].biases = biases0_all[..., idx_neurons]
    model[1].weights = np.row_stack(weights)
    model[1].biases = np.sum(biases, axis=0)
