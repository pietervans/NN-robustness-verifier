import numpy as np
from numpy import ndarray, logical_or, logical_and
from numpy.core.numeric import full_like, zeros_like


# TODO keep this list up to date with all sound heuristics
ALL_HEURISTICS = [
    'min_area', 
    'min_area2', 
    'alt_crossing_area', 
    'only_feasible', 
    'min_worst_case',
    'specified_lower_0.1',
    'specified_lower_0.25',
    'specified_lower_0.35',
    'specified_lower_0.5',
]

# NOTE heuristics that are tried first (in combinations) because they are most promising
BEST_HEURISTICS = [
    'only_feasible',
    'specified_lower_0.1',
    'specified_lower_0.25',
    'specified_lower_0.45'
]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SPU_activation(x: ndarray):
    result = zeros_like(x)
    x_below_zero = x < 0
    result[x_below_zero] = 1/(1+np.exp(x[x_below_zero])) - 1
    x_above_zero = ~x_below_zero
    result[x_above_zero] = x[x_above_zero]**2 - 0.5
    return result

def SPU_left_derivative(x):
    exp_x = np.exp(x)
    return -exp_x/(1+exp_x)**2

def SPU_left_derivative_inverse(x):
    '''Derivative inverse of the left side (x < 0) of the SPU function'''
    x_2 = 2*x
    return np.log((-x_2 + np.sqrt(4*x + 1) - 1) / x_2)

def SPU_left_derivative_inverse_exp(x):
    '''Exponentiated derivative inverse of the left side (x < 0) of the SPU function.
    Used to simplify some calculations.'''
    nonzero_inds = x != 0
    x_2 = 2*x
    result = full_like(x, -np.inf)
    result[nonzero_inds] = (-x_2[nonzero_inds] + np.sqrt(4*x[nonzero_inds] + 1) - 1) / x_2[nonzero_inds]
    return result

def protected_log(x):
    result = full_like(x, -np.inf)
    nonzero_inds = x != 0
    result[nonzero_inds] = np.log(x[nonzero_inds])
    return result

def tangent_at(l: ndarray, u: ndarray, SPU_l: ndarray, t: float):
    '''Returns the tangent of SPU a fraction along the interval [l, u].
    If the position is negative, draw a secant between [l, 0] instead'''

    w_l, b_l = zeros_like(l), zeros_like(l)
    position = l + t * (u - l)
    tangent_inds = position >= 0
    secant_inds = ~tangent_inds
    w_l[tangent_inds] = 2 * position[tangent_inds]
    b_l[tangent_inds] = -position[tangent_inds] ** 2 - 0.5
    w_l[secant_inds] = (0.5 + SPU_l[secant_inds]) / l[secant_inds] 
    b_l[secant_inds] = -0.5
    return w_l, b_l


def compute_interval_bounds(l: ndarray, u: ndarray):
    '''Determine the smallest interval [l_out, u_out] that captures the output range'''
    l_out = np.zeros_like(l)
    u_out = np.zeros_like(u)

    # If l >= 0
    l_above_zero = l >= 0
    l_out[l_above_zero] = l[l_above_zero]**2 - 0.5
    u_out[l_above_zero] = u[l_above_zero]**2 - 0.5

    # else, if u <= 0
    u_below_zero = u <= 0
    l_out[u_below_zero] = sigmoid(-u[u_below_zero]) - 1
    u_out[u_below_zero] = sigmoid(-l[u_below_zero]) - 1

    # else
    crossing_case = np.logical_and(~l_above_zero, ~u_below_zero)
    l_out[crossing_case] = -0.5

    # If the shape is not (n, 1) and instead (n, ) -- slightly more computations
    if len(crossing_case.shape) == 1:
        n_crossing_case = crossing_case.sum()
        u_out[crossing_case] = np.max(np.append(sigmoid(-l[crossing_case].reshape(n_crossing_case, 1)) - 1,
                                    (u[crossing_case]**2 - 0.5).reshape(n_crossing_case, 1), axis=1), axis=1)
    else:
        u_out[crossing_case] = np.max(np.append(sigmoid(-l[crossing_case[:, 0], :]) - 1,
                                     (u[crossing_case[:, 0], :]**2 - 0.5), axis=1), axis=1)
    
    return l_out, u_out


def compute_relational_bounds(l, u, heuristic):
    '''Determine the upper and lower bounding lines according to the given heuristic'''
    if heuristic == 'min_area':
        return compute_bounds_min_area(l, u)
    elif heuristic == 'min_area2':
        return compute_bounds_min_area2(l, u)
    elif heuristic == 'alt_crossing_area':
        return compute_bounds_alt_crossing_area(l, u)
    elif heuristic[:15] == 'specified_lower':
        t = float(heuristic[16:])
        return compute_bounds_specified_lower(l, u, t)
    elif heuristic == 'only_feasible':
        return compute_bounds_only_feasible(l, u)
    elif heuristic == 'min_worst_case':
        return compute_bounds_min_worst_case(l, u)
    raise Exception('Enter a valid heuristic')


def compute_bounds_min_area(l: ndarray, u: ndarray):
    '''Compute the linear bounds that result in minimal area between the lines'''

    w_u, b_u, w_l, b_l = zeros_like(u), zeros_like(u), zeros_like(l), zeros_like(l)
    exp_l = np.exp(l)

    above_zero = l >= 0
    w_u[above_zero] = u[above_zero] + l[above_zero]
    b_u[above_zero] = -u[above_zero] * l[above_zero] - 0.5
    w_l[above_zero] = w_u[above_zero]
    b_l[above_zero] = -w_l[above_zero]**2 / 4 - 0.5

    below_zero = u <= 0
    sigm_minus_l = 1 / (1 + exp_l[below_zero])
    sigm_minus_u = sigmoid(-u[below_zero])

    # NOTE The tangent line for input=l is used as the upper bound (it's unproven that this is optimal).
    w_u[below_zero] = -exp_l[below_zero]/(1+exp_l[below_zero])**2
    b_u[below_zero] = sigm_minus_l - 1 - w_u[below_zero] * l[below_zero]
    w_l[below_zero] = (sigm_minus_u - sigm_minus_l) / (u[below_zero] - l[below_zero])
    b_l[below_zero] = sigm_minus_u - 1 - u[below_zero] * w_l[below_zero]

    # These cases are similar to a crossing ReLU
    # NOTE the trivial upper bounding line (connecting points) could be unsound in this case (credits to Mika)
    # We check if the trivial line would result in an unsound transformer by drawing the tangent
    # for input=l, determining the intersection with the quadratic part and comparing this with u.
    # If u is larger than the intersection, then the trivial line is sound.

    crossing = np.logical_and(~below_zero, ~above_zero)

    w_tangent_l = -exp_l / (1+exp_l) ** 2
    sru_left = 1 / (1 + exp_l) - 1
    b_tangent_l = sru_left - w_tangent_l * l
    intersection_with_quadratic = np.full_like(u, -np.inf)
    intersection_with_quadratic[crossing] = 0.5*(w_tangent_l[crossing] + 
                    np.sqrt(w_tangent_l[crossing]**2 + 4*(b_tangent_l[crossing] + 0.5)))

    above_intersect = np.logical_and(crossing, u > intersection_with_quadratic)
    l_above_intersect = l[above_intersect]
    u_above_intersect = u[above_intersect]

    w_u[above_intersect] = (u_above_intersect**2 + 0.5 - sigmoid(-l_above_intersect)) / \
        (u_above_intersect - l_above_intersect)
    b_u[above_intersect] = u_above_intersect**2 - 0.5 - u_above_intersect * w_u[above_intersect]

    below_intersect = np.logical_and(crossing, u <= intersection_with_quadratic)

    # TODO upper bound not optimal
    # Now, we use the tangent in input=l as an approximation
    w_u[below_intersect] = w_tangent_l[below_intersect]
    b_u[below_intersect] = b_tangent_l[below_intersect]

    u_greater = np.logical_and(u > -l, crossing)
    l_greater = np.logical_and(u <= -l, crossing)
    w_l[u_greater] = u[u_greater] + l[u_greater]
    b_l[u_greater] = -w_l[u_greater]**2 / 4 - 0.5
    w_l[l_greater] = 1/l[l_greater] * (1 / (1+exp_l[l_greater]) - 0.5)
    b_l[l_greater] = -0.5
    
    return (w_l, b_l), (w_u, b_u)


def compute_bounds_min_area2(l: ndarray, u: ndarray):
    '''Compute the linear bounds that result in minimal area between the lines.
    This version also implements the inverse left derivative for tighter bounds.'''

    SPU_l = SPU_activation(l)
    SPU_u = SPU_activation(u)
    w_u = (SPU_u - SPU_l) / (u - l)
    b_u = SPU_u - w_u * u
    w_l, b_l = zeros_like(l), zeros_like(l)
    exp_l = np.exp(l)

    above_zero = l >= 0
    w_l[above_zero] = w_u[above_zero]
    b_l[above_zero] = -w_l[above_zero]**2 / 4 - 0.5

    below_zero = u <= 0
    sigm_minus_l = 1 / (1 + exp_l[below_zero])
    sigm_minus_u = sigmoid(-u[below_zero])

    # NOTE The tangent line for input=l is used as the upper bound (it's unproven that this is optimal).
    w_l[below_zero] = (sigm_minus_u - sigm_minus_l) / (u[below_zero] - l[below_zero])
    b_l[below_zero] = sigm_minus_u - 1 - u[below_zero] * w_l[below_zero]

    # These cases are similar to a crossing ReLU
    # NOTE the trivial upper bounding line (connecting points) could be unsound in this case (credits to Mika)
    # We check if the trivial line would result in an unsound transformer by drawing the tangent
    # for input=l, determining the intersection with the quadratic part and comparing this with u.
    # If u is larger than the intersection, then the trivial line is sound.

    crossing = np.logical_and(~below_zero, ~above_zero)
    u_greater0 = u > 0
    w_u_greater0 = w_u >= 0
    
    sigmoid_case = ~w_u_greater0
    derivative_smaller = SPU_left_derivative(l) <= w_u
    secant_case = logical_and(sigmoid_case, derivative_smaller)
    b_u[secant_case] = SPU_u[secant_case] -w_u[secant_case] * u[secant_case]

    # If the derivative of SPU is greater, we have to make a tangent of SPU and w_u
    tangent_case = logical_and(sigmoid_case, ~derivative_smaller)
    exp_x = SPU_left_derivative_inverse_exp(w_u[tangent_case])
    x = protected_log(exp_x)
    b_u[tangent_case] = 1 / (1 + exp_x) - 1 - w_u[tangent_case] * x

    # Check instead if everything is below zero
    below_zero = np.argwhere(~u_greater0.flatten()).flatten()
    exp_x = SPU_left_derivative_inverse_exp(w_u[below_zero])

    # If the slope w_u is close to zero, things get messed up,
    # so we need to protect against this
    zero_slope = np.argwhere(exp_x.flatten() == 0)
    zero_slope = np.append(zero_slope, np.argwhere(abs(exp_x.flatten()) == np.inf))
    derivative_l = SPU_left_derivative(l[below_zero[zero_slope]])
    w_u[below_zero[zero_slope]] = derivative_l
    b_u[below_zero[zero_slope]] = SPU_l[below_zero[zero_slope]] - derivative_l * l[below_zero[zero_slope]]

    u_greater = np.logical_and(u > -l, crossing)
    l_greater = np.logical_and(u <= -l, crossing)
    w_l[u_greater] = u[u_greater] + l[u_greater]
    b_l[u_greater] = -w_l[u_greater]**2 / 4 - 0.5
    w_l[l_greater] = 1/l[l_greater] * (1 / (1+exp_l[l_greater]) - 0.5)
    b_l[l_greater] = -0.5
    
    return (w_l, b_l), (w_u, b_u)


def compute_bounds_alt_crossing_area(l: ndarray, u: ndarray):
    '''Compute the linear bounds that result in minimal area between the lines.
    This variant chooses different bounds in the crossing region, however, not leading to minimal area.'''

    SPU_l = SPU_activation(l)
    SPU_u = SPU_activation(u)
    w_u = (SPU_u - SPU_l) / (u - l)
    b_u = SPU_u - w_u * u
    w_l, b_l = zeros_like(l), zeros_like(l)
    exp_l = np.exp(l)

    above_zero = l >= 0
    w_l[above_zero] = w_u[above_zero]
    b_l[above_zero] = -w_l[above_zero]**2 / 4 - 0.5

    below_zero = u <= 0
    sigm_minus_l = 1 / (1 + exp_l[below_zero])
    sigm_minus_u = sigmoid(-u[below_zero])

    # NOTE The tangent line for input=l is used as the upper bound (it's unproven that this is optimal).
    w_l[below_zero] = (sigm_minus_u - sigm_minus_l) / (u[below_zero] - l[below_zero])
    b_l[below_zero] = sigm_minus_u - 1 - u[below_zero] * w_l[below_zero]

    # These cases are similar to a crossing ReLU
    # NOTE the trivial upper bounding line (connecting points) could be unsound in this case (credits to Mika)
    # We check if the trivial line would result in an unsound transformer by drawing the tangent
    # for input=l, determining the intersection with the quadratic part and comparing this with u.
    # If u is larger than the intersection, then the trivial line is sound.

    crossing = np.logical_and(~below_zero, ~above_zero)
    u_greater0 = u > 0
    w_u_greater0 = w_u >= 0
    
    sigmoid_case = ~w_u_greater0
    derivative_smaller = SPU_left_derivative(l) <= w_u
    secant_case = logical_and(sigmoid_case, derivative_smaller)
    b_u[secant_case] = SPU_u[secant_case] -w_u[secant_case] * u[secant_case]

    # If the derivative of SPU is greater, we have to make a tangent of SPU and w_u
    tangent_case = logical_and(sigmoid_case, ~derivative_smaller)
    exp_x = SPU_left_derivative_inverse_exp(w_u[tangent_case])
    x = protected_log(exp_x)
    b_u[tangent_case] = 1 / (1 + exp_x) - 1 - w_u[tangent_case] * x

    # Check instead if everything is below zero
    below_zero = np.argwhere(~u_greater0.flatten()).flatten()
    exp_x = SPU_left_derivative_inverse_exp(w_u[below_zero])

    # If the slope w_u is close to zero, things get messed up,
    # so we need to protect against this
    zero_slope = np.argwhere(exp_x.flatten() == 0)
    zero_slope = np.append(zero_slope, np.argwhere(abs(exp_x.flatten()) == np.inf))
    derivative_l = SPU_left_derivative(l[below_zero[zero_slope]])
    w_u[below_zero[zero_slope]] = derivative_l
    b_u[below_zero[zero_slope]] = SPU_l[below_zero[zero_slope]] - derivative_l * l[below_zero[zero_slope]]

    u_greater = np.logical_and(u > -l, crossing)
    l_greater = np.logical_and(u <= -l, crossing)
    w_l[l_greater] = u[l_greater]
    b_l[l_greater] = -w_l[l_greater]**2 / 4 - 0.5
    w_l[u_greater] = 1/l[u_greater] * (1 / (1+exp_l[u_greater]) - 0.5)
    b_l[u_greater] = -0.5
    
    return (w_l, b_l), (w_u, b_u)


def compute_bounds_specified_lower(l: ndarray, u: ndarray, t: float = 0.5):
    '''Same as min_area2, but the lower bound is given by the tangent
    at a specified fraction along l and u. '''

    SPU_l = SPU_activation(l)
    SPU_u = SPU_activation(u)
    w_u = (SPU_u - SPU_l) / (u - l)
    b_u = SPU_u - w_u * u
    w_l, b_l = zeros_like(l), zeros_like(l)
    exp_l = np.exp(l)

    above_zero = l >= 0
    below_zero = u <= 0
    sigm_minus_l = 1 / (1 + exp_l[below_zero])
    sigm_minus_u = sigmoid(-u[below_zero])

    # NOTE The tangent line for input=l is used as the upper bound (it's unproven that this is optimal).
    w_l[below_zero] = (sigm_minus_u - sigm_minus_l) / (u[below_zero] - l[below_zero])
    b_l[below_zero] = sigm_minus_u - 1 - u[below_zero] * w_l[below_zero]

    # These cases are similar to a crossing ReLU
    # NOTE the trivial upper bounding line (connecting points) could be unsound in this case (credits to Mika)
    # We check if the trivial line would result in an unsound transformer by drawing the tangent
    # for input=l, determining the intersection with the quadratic part and comparing this with u.
    # If u is larger than the intersection, then the trivial line is sound.

    u_greater0 = u > 0
    w_u_greater0 = w_u >= 0
    w_l[u_greater0], b_l[u_greater0] = tangent_at(l[u_greater0], u[u_greater0], SPU_l[u_greater0], t)

    sigmoid_case = ~w_u_greater0
    derivative_smaller = SPU_left_derivative(l) <= w_u
    secant_case = logical_and(sigmoid_case, derivative_smaller)
    b_u[secant_case] = SPU_u[secant_case] -w_u[secant_case] * u[secant_case]

    # If the derivative of SPU is greater, we have to make a tangent of SPU and w_u
    tangent_case = logical_and(sigmoid_case, ~derivative_smaller)
    exp_x = SPU_left_derivative_inverse_exp(w_u[tangent_case])
    x = protected_log(exp_x)
    b_u[tangent_case] = 1 / (1 + exp_x) - 1 - w_u[tangent_case] * x

    # Check instead if everything is below zero
    below_zero = np.argwhere(~u_greater0.flatten()).flatten()
    exp_x = SPU_left_derivative_inverse_exp(w_u[below_zero])

    # If the slope w_u is close to zero, things get messed up,
    # so we need to protect against this
    zero_slope = np.argwhere(exp_x.flatten() == 0)
    zero_slope = np.append(zero_slope, np.argwhere(abs(exp_x.flatten()) == np.inf))
    derivative_l = SPU_left_derivative(l[below_zero[zero_slope]])
    w_u[below_zero[zero_slope]] = derivative_l
    b_u[below_zero[zero_slope]] = SPU_l[below_zero[zero_slope]] - derivative_l * l[below_zero[zero_slope]]

    return (w_l, b_l), (w_u, b_u)

def compute_bounds_only_feasible(l: ndarray, u: ndarray):
    '''Compute the bounds which minimise the area
    under the constraint such that no unreachable SPU values
    for the interval are included between the lines.'''

    SPU_l = SPU_activation(l)
    SPU_u = SPU_activation(u)
    w_l, w_u, b_l, b_u = zeros_like(l), full_like(u, -np.inf), zeros_like(l), zeros_like(u)

    # Check first if we have crossing cases 
    crossing = logical_and(l <= 0, u > 0)
    w_l[crossing] = 0
    b_l[crossing] = -0.5
    w_u[crossing] = (SPU_u[crossing] - SPU_l[crossing]) / (u[crossing] - l[crossing])

    # Check if the value of SPU(u) > SPU(l)
    upper_greater = SPU_u > SPU_l
    derivative_smaller = SPU_left_derivative(l) <= w_u

    # If SPU(u) > SPU(l) and the derivative is smaller than w_u, then we can make a tangent
    tangent_case = logical_and(crossing, logical_or(upper_greater, derivative_smaller))
    b_u[tangent_case] = SPU_u[tangent_case] - w_u[tangent_case] * u[tangent_case]

    # If it is not, set the derivative to the one at the left endpoint
    unsound_case = logical_and(crossing, logical_and(~upper_greater, ~derivative_smaller))
    derivative_l = SPU_left_derivative(l[unsound_case])
    w_u[unsound_case] = derivative_l
    b_u[unsound_case] = SPU_l[unsound_case] - derivative_l * l[unsound_case]

    # Check if everything is below 0
    below = u <= 0
    derivative_l = SPU_left_derivative(l[below])
    w_u[below] = derivative_l
    b_u[below] = SPU_l[below] - derivative_l * l[below]
    w_l[below] = (SPU_u[below] - SPU_l[below]) / (u[below] - l[below])
    b_l[below] = SPU_u[below] - w_l[below] * u[below]

    # Check if everything is above 0
    above = l >= 0
    w_u[above] = (SPU_u[above] - SPU_l[above]) / (u[above] - l[above])
    b_u[above] = SPU_u[above] - w_u[above] * u[above]
    w_l[above] = 2*l[above]
    b_l[above] = SPU_l[above] - w_l[above] * l[above]

    return (w_l, b_l), (w_u, b_u)


def compute_bounds_min_worst_case(l: ndarray, u: ndarray):
    '''This minimises the worst-case distance,
    while not necessarily choosing the minimum-area solution.'''

    SPU_l = SPU_activation(l)
    SPU_u = SPU_activation(u)

    w_u = (SPU_u - SPU_l) / (u - l)
    w_l = np.copy(w_u)
    b_l, b_u = zeros_like(l), zeros_like(u)

    # Check if u is greater than 0
    u_greater0 = u >= 0
    b_u[u_greater0] = SPU_u[u_greater0] - w_u[u_greater0] * u[u_greater0]

    # If it is, check if the slope of upper line is greater than 0
    w_u_greater0 = w_u >= 0
    quadratic_case = logical_and(u_greater0, w_u_greater0)
    b_l[quadratic_case] = -1/4 * (2 + w_l[quadratic_case] ** 2)

    # If the slope is lower than 0, then we have to control soundness
    sigmoid_case = logical_and(u_greater0, ~w_u_greater0)
    b_l[sigmoid_case] = -0.5
    w_l[sigmoid_case] = (0.5 + SPU_l[sigmoid_case]) / l[sigmoid_case]

    derivative_smaller = SPU_left_derivative(l) <= w_u
    secant_case = logical_and(sigmoid_case, derivative_smaller)
    b_u[secant_case] = SPU_u[secant_case] -w_u[secant_case] * u[secant_case]

    # If the derivative of SPU is greater, we have to make a tangent of SPU and w_u
    tangent_case = logical_and(sigmoid_case, ~derivative_smaller)
    exp_x = SPU_left_derivative_inverse_exp(w_u[tangent_case])
    x = protected_log(exp_x)
    b_u[tangent_case] = 1 / (1 + exp_x) - 1 - w_u[tangent_case] * x

    # Check instead if everything is below zero
    below_zero = np.argwhere(~u_greater0.flatten()).flatten()
    exp_x = SPU_left_derivative_inverse_exp(w_u[below_zero])

    # If the slope w_u is close to zero, things get messed up,
    # so we need to protect against this
    zero_slope = np.argwhere(exp_x.flatten() == 0)
    zero_slope = np.append(zero_slope, np.argwhere(abs(exp_x.flatten()) == np.inf))
    derivative_l = SPU_left_derivative(l[below_zero[zero_slope]])
    w_u[below_zero[zero_slope]] = derivative_l
    b_u[below_zero[zero_slope]] = SPU_l[below_zero[zero_slope]] - derivative_l * l[below_zero[zero_slope]]
    exp_x = np.delete(exp_x, zero_slope, 0)
    below_zero = np.delete(below_zero, zero_slope, 0)

    x = protected_log(exp_x)
    b_u[below_zero] = 1 / (1 + exp_x) - 1 - w_u[below_zero] * x
    b_l[below_zero] = SPU_u[below_zero] - w_l[below_zero] * u[below_zero]

    return (w_l, b_l), (w_u, b_u)
