import numpy as np
from transformers import ALL_HEURISTICS, SPU_activation, compute_interval_bounds, compute_relational_bounds


def test_plot(heuristic):
    import matplotlib.pyplot as plt
    l, u = np.array([np.random.uniform(-4, 2)]), np.array([np.random.uniform(-2, 4)])
    if l[0] > u[0]:
        l, u = u, l
    (w_l, b_l), (w_u, b_u) = compute_relational_bounds(l, u, heuristic)

    l, u = l[0], u[0]
    (w_l, b_l), (w_u, b_u) = (w_l[0], b_l[0]), (w_u[0], b_u[0])
    x = np.linspace(l-.2, u+.2, 200)
    x_neg = x[x < 0]
    x_pos = x[x >= 0]
    plt.plot(x_neg, 1/(1+np.exp(x_neg)) - 1, 'b')
    plt.plot(x_pos, np.power(x_pos, 2)-0.5, 'b')
    plt.plot(x, w_l*x + b_l, '--')
    plt.plot(x, w_u*x + b_u, '--')
    plt.plot(l, SPU_activation(l), 'o', color='k', markersize=10)
    plt.plot(u, SPU_activation(u), 'o', color='k', markersize=10)
    plt.show()


def verify_interval_bounds():

    l, u = np.random.uniform(-5, 3, size=(300,1)), np.random.uniform(-3, 5, size=(300,1))
    switch = l > u
    temp = u[switch]
    u[switch] = l[switch]
    l[switch] = temp

    l_out, u_out = compute_interval_bounds(l, u)
    vals = np.linspace(l, u, 50)
    SPU_vals = SPU_activation(vals)
    
    assert np.all(SPU_vals >= l_out - 1e-14)
    assert np.all(SPU_vals <= u_out + 1e-14)


def verify(heuristic):
    '''Verify that the transformer for the given heuristic is sound'''
    l, u = np.random.uniform(-5, 3, size=(1000,1)), np.random.uniform(-3, 5, size=(1000,1))
    switch = l > u
    temp = u[switch]
    u[switch] = l[switch]
    l[switch] = temp

    (w_l, b_l), (w_u, b_u) = compute_relational_bounds(l, u, heuristic)
    
    vals = np.linspace(l, u, 200)
    SPU_vals = SPU_activation(vals)
    
    assert np.all(SPU_vals >= w_l*vals + b_l - 1e-14)
    assert np.all(SPU_vals <= w_u*vals + b_u + 1e-14)


if __name__ == '__main__':

    # a = 'random'
    # while a != 'c':
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     test_plot('min_worst_case')
    #     a = input('Press c to stop plotting')

    print('Testing interval bounds...')
    verify_interval_bounds()
    print('compute_interval_bounds is sound!')

    for heuristic in ALL_HEURISTICS:
        print(f'Testing heuristic: {heuristic}')
        verify(heuristic)
        print('Test successful!')
