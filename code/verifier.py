import argparse
import torch
from networks import FullyConnected
from transformers import compute_relational_bounds, ALL_HEURISTICS, BEST_HEURISTICS
import numpy as np
import random
import itertools

DEVICE = 'cpu'
INPUT_SIZE = 28


def try_combinations(subset_heuristics, nb_SPU_layers, net, inputs, eps, true_label):
    '''Try all combinations for the given subset (nb_heuristics**nb_SPU_layers of them)'''

    all_combinations = list(itertools.product(subset_heuristics, repeat=nb_SPU_layers))
    random.shuffle(all_combinations)
    for combination in all_combinations:
        verified = analyze_with_heuristics(combination, net, inputs, eps, true_label)
        if verified:
            return verified
    return False


def analyze(net, inputs: torch.Tensor, eps: float, true_label: int):
    '''Return true iff verified (with any combination of heuristics)'''

    nb_SPU_layers = int((len(net.layers)-3)/2)
    for heuristic in ALL_HEURISTICS:
        verified = analyze_with_heuristics((heuristic,)*nb_SPU_layers, net, inputs, eps, true_label)
        if verified:
            return verified

    # Idea: using a random heuristic in every layer of the network can result in verification
    # We try out every possible combination
    
    for subset_heuristics in [BEST_HEURISTICS, ALL_HEURISTICS]:
        verified = try_combinations(subset_heuristics, nb_SPU_layers, net, inputs, eps, true_label)
        if verified:
            return verified
    
    return False


def analyze_with_heuristics(heuristics, net, inputs: torch.Tensor, eps: float, true_label: int):
    '''Return true iff verified with the given sequence of heuristics'''

    # net.layers = [Normalization, Flatten, Linear, SPU, ..., Linear]
    normalization_layer = net.layers[0]
    mean = normalization_layer.mean.item()
    sigma = normalization_layer.sigma.item()

    # List with all weights and biases of each linear layer
    parameters = []
    for i in range(2, len(net.layers), 2):
        linear_layer = net.layers[i]
        weights = linear_layer.weight.detach().numpy()                  # Dim = #outputs x #inputs
        biases = np.reshape(linear_layer.bias.detach().numpy(), (-1,1)) # Dim = #outputs x 1
        weights_and_biases = np.concatenate((weights, biases), axis=1)  # Dim = #outputs x #inputs+1
        parameters.append(weights_and_biases)

    # E.g. for fc1: 1 hidden layer with 50 neurons (784 -> 50 -> 10)
    # parameters has two elements: 1) 50 x (784+1), 2) 10 x (50+1)

    inputs = inputs.detach().numpy() # Dim = (1,1,28,28)
    inputs = np.reshape(inputs, (784, -1))

    # Create box for inputs
    inputs_box = np.hstack((np.where(inputs-eps > 0, inputs-eps, 0), np.where(inputs+eps<1, inputs+eps, 1))) # Dim = 784 x 2

    # Normalization
    norm_inputs_box = (inputs_box-mean)/sigma

    # Appending [1,1] as a virtual neuron (makes handling of biases easier)
    norm_inputs_box = np.vstack([norm_inputs_box, [1,1]]) # Dim = (784+1) x 2

    rel_l_matrices = [] # List of matrices with relational lower bounds
    rel_u_matrices = [] # List of matrices with relational upper bounds

    nb_of_linear_layers = int((len(net.layers)-1)/2)
    for lin_index in range(nb_of_linear_layers):

        # LINEAR LAYER
        wb = parameters[lin_index] # weights and biases of linear layer = relational upper and lower bounds

        # Add row for virtual neuron
        wb = add_row_virtual_neuron(wb)

        # The relational upper and lower bounds are exact for affine transform
        rel_l_matrices.append(wb)
        rel_u_matrices.append(wb)

        if lin_index == nb_of_linear_layers-1:
            # We processed the last linear layer, so there's no SPU layer following this layer
            break

        # Determine interval bounds
        interval_l, interval_u = backsubstitute(2*lin_index-1, wb, wb, rel_l_matrices, rel_u_matrices, norm_inputs_box)

        # SPU LAYER
        # We do not include the virtual neuron in the arguments
        (w_l, b_l), (w_u, b_u) = compute_relational_bounds(interval_l[:-1], interval_u[:-1], heuristics[lin_index])

        # The matrices representing the relational bounds (analogous to the matrices for a linear layer)
        wb_l = get_relational_bounds_matrix(w_l, b_l)
        wb_u = get_relational_bounds_matrix(w_u, b_u)
        rel_l_matrices.append(wb_l)
        rel_u_matrices.append(wb_u)

        # Interval bounds can be determined as follows (but we do not need them here)
        # interval_l, interval_u = backsubstitute(2*lin_index, wb_l, wb_u, rel_l_matrices, rel_u_matrices, norm_inputs_box)


    # Now any array of expressions written in terms of the neurons of the last layer can be backsubstituted through the network
    # For instance, to compute the lower and upper bounds for  2*last_layer[0] - last_layer[9] + 3
    # expr = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, -1, 3]])
    # l, u = backsubstitute(len(rel_l_matrices)-1, expr, expr, rel_l_matrices, rel_u_matrices, norm_inputs_box)

    # VERIFICATION
    # Check that the activation value for the true label
    # is strictly greater than all other activation values in the last layer
    
    # Differences of activations in last layer: (true - any other) --> 9 expressions
    differences = np.hstack((-np.eye(10), np.zeros((10,1))))
    differences = np.delete(differences, true_label, 0)
    differences[:,true_label] = 1

    lower_bounds = backsubstitute_lower(len(rel_l_matrices)-1, differences, rel_l_matrices, rel_u_matrices, norm_inputs_box)
    
    return np.all((lower_bounds > 0) == 1)


def backsubstitute(index, wb_l, wb_u, rel_l_matrices, rel_u_matrices, norm_inputs_box):
    '''Return the lower and upper bounds according to the relative bounds wb_l and wb_u.
    param index = the index of the relative bound matrix of the previous layer (in the rel_[l|u]_matrices).'''
    return backsubstitute_lower(index, wb_l, rel_l_matrices, rel_u_matrices, norm_inputs_box), \
           backsubstitute_upper(index, wb_u, rel_l_matrices, rel_u_matrices, norm_inputs_box)


def backsubstitute_lower(index, wb_l, rel_l_matrices, rel_u_matrices, norm_inputs_box):
    wb_l_pos = np.where(wb_l > 0, wb_l, 0)
    wb_l_neg = np.where(wb_l < 0, wb_l, 0)
    if index < 0:
        input_l = norm_inputs_box[:,0]
        input_u = norm_inputs_box[:,1]
        wb_l_new = wb_l_pos @ input_l + wb_l_neg @ input_u
        return wb_l_new
    wb_l_new = wb_l_pos @ rel_l_matrices[index] + wb_l_neg @ rel_u_matrices[index]
    return backsubstitute_lower(index-1, wb_l_new, rel_l_matrices, rel_u_matrices, norm_inputs_box)
    

def backsubstitute_upper(index, wb_u, rel_l_matrices, rel_u_matrices, norm_inputs_box):
    wb_u_pos = np.where(wb_u > 0, wb_u, 0)
    wb_u_neg = np.where(wb_u < 0, wb_u, 0)
    if index < 0:
        input_l = norm_inputs_box[:,0]
        input_u = norm_inputs_box[:,1]
        wb_u_new = wb_u_pos @ input_u + wb_u_neg @ input_l
        return wb_u_new
    wb_u_new = wb_u_pos @ rel_u_matrices[index] + wb_u_neg @ rel_l_matrices[index]
    return backsubstitute_upper(index-1, wb_u_new, rel_l_matrices, rel_u_matrices, norm_inputs_box)


def add_row_virtual_neuron(a):
    '''Add a row to matrix a for a virtual neuron. This row is [0 0 ... 0 1]'''
    row = np.zeros((1, np.size(a, 1)))
    row[0, -1] = 1
    return np.vstack([a, row])

def get_relational_bounds_matrix(slopes, intercepts):
    '''Return a (sparse) matrix that represents the relational bounds'''
    result = np.diag(slopes)
    result = np.hstack([result, np.reshape(intercepts, (-1,1))])
    return add_row_virtual_neuron(result)


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
