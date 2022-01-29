import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
import os
import random
import scipy

# ---------------------------------------------------------------------
# setting
# ---------------------------------------------------------------------
def samples_file(hyper_paras, info=""):
    # image_size = hyper_paras['image_size']
    M_samples_per_para = hyper_paras['M_samples_per_para']
    sigma_range = hyper_paras['sigma_range']
    noise_num = hyper_paras['noise_num']
    data_base = 'data'
    if not os.path.exists(data_base):
        os.makedirs(data_base)
    file = os.path.join(data_base,
                        f'{info}'
                        f'_mu{noise_num}_{sigma_range[0]}_{sigma_range[1]}'
                        f'_M{M_samples_per_para}')
    return file


def get_x_ml(y,hypers):
    """
    max p(y|x)
    """
    # y is  a ndarray
    n,m = hypers['data_dim'],hypers['x_dim']
    data = y.flatten()
    # Construct the problem.

    x = cp.Variable([m,1])
    f = cp.sum_squares(data-x.flatten())
    objective = cp.Minimize(f)
    p = cp.Problem(objective)

    # Assign a value to gamma and find the optimal x.
    result = p.solve(solver=cp.SCS,verbose=False)
    return (x.value).reshape(-1,1)