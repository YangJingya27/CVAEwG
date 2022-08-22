import pandas as pd
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar
import os
import random
import scipy

m = 26
n = 50
alpha = 0.01
beta = 0.0001
H = np.array(pd.read_csv('H.csv'))
Y = np.array(pd.read_csv('y.csv')) # 之前模拟产生的数据
N = 10000


def W_matrx_1():
    W = np.zeros([m,m])
    for i in range(m):
        if i ==0:
           W[i,i] = 1
           W[i,i+1] = -1
        elif i == m-1:
           W[i, i] = 1
           W[i, i - 1] = -1
        else:
            W[i, i] = 2
            W[i, i + 1] = -1
            W[i, i - 1] = -1
    return W

def W_matrx_2():
    W = np.identity(m)
    return W


W = W_matrx_2()


# p(theta) 先验分布,theta.shape为m*1
def p_theta(theta,sigma):
    lamda = 2*alpha/sigma**2
    p = lamda**(m/2)*np.exp(-0.5*lamda*theta.T@W@theta)
    return p

#---------------------------------------------------------------------------------
# 优化部分
#------------------------------------------------------------------------------------
# p(sigma|theta,Y),后验分布,用于在gibbs抽样之前用MAP确定sigma
# def post_sigma(sigma,theta,Y):
#     C = (H@theta-Y).T@(H@theta-Y)+2*alpha *theta.T@W@theta
#     p = sigma**(-(n+m))*np.exp(-C/(2*sigma**2)-sigma**2/beta)
#     return p[0][0]

def get_sigma_map(theta,Y):
    C = (H @ theta - Y).T @ (H @ theta - Y) + 2 * alpha * theta.T @ W @ theta
    f = lambda x: (n+m)*np.log(x) +C/(2*x**2)+x**2/beta
    res = minimize_scalar(f, method='brent')
    return res.x

# p(theta|sigma,Y) 后验分布，用于在确定sigma后用MAP确定初始值
# def post_theta(theta,sigma,Y,W):
#     HT_H = (H@theta-Y).T@(H@theta-Y)
#     HT_H = HT_H[0][0]
#     TWT = theta.T@ W @theta
#     TWT = TWT[0][0]
#     p1 = 1/(2*sigma**2)* HT_H
#     p2 = alpha/sigma**2 * TWT
#     a = np.exp(-(p1+p2))
#     return a # 负指数部分
def get_x_ml(y,hypers):
    """
    max p(y|x)
    """
    # y is  a ndarray
    n,m = hypers['data_dim'],hypers['x_dim']
    H = np.array(hypers['H'])

    data = y.flatten()
    # Construct the problem.

    x = cp.Variable([m,1])
    f = cp.sum_squares(data-H@x.flatten())
    objective = cp.Minimize(f)
    p = cp.Problem(objective)

    # Assign a value to gamma and find the optimal x.
    result = p.solve(solver=cp.ECOS,verbose=False)
    return (x.value).reshape(-1,1)


def get_theta_map(sigma,Y):
    """
    max p(theta|y)
    """
    I = np.identity(n)
    # Construct the problem.
    x = cp.Variable((m, 1))
    f = 1/(2*sigma**2)*cp.quad_form(H@x-Y,I) + alpha/sigma**2 * cp.quad_form(x,W)
    objective = cp.Minimize(f)
    p = cp.Problem(objective)
    # Assign a value to gamma and find the optimal x.
    result = p.solve(solver=cp.SCS,verbose=False)
    return (x.value).reshape(-1,1)

# _____________________________________________________________________________________
# gibbs sampling
# ----------------------------------------------------------------------------------
def prepare_(i, theta, H, Y, sigma):
    n = 50
    lamda = 2 * alpha / sigma ** 2
    index = [k for k in range(26)]
    index.remove(i)
    a_i_r = 0
    b_i_r = 0
    for s in range(n):
        mu_s_r = 0
        for j in index:
            # mu_p_l += W[j,i]*theta[j]
            # mu_p_r += W[i,j]*theta[j]
            mu_s_r += H[s,j]*theta[j]
        mu_p = 0
        a_i_r += (H[s,i]**2) / (sigma**2)
        mu_s = Y[s] - mu_s_r
        b_i_r += 2*mu_s*H[s,i]/(sigma**2)
    a_i = a_i_r+lamda*W[i,i]
    b_i = b_i_r-lamda*mu_p
    mu_i = b_i/(2*a_i)
    sigma_i = a_i ** (-0.5)
    return mu_i[0], sigma_i


def get_theta(mu_i, sigma_i):
    theta = mu_i + sigma_i**2 * np.random.randn(1)
    return theta[0]

# ---------------------------------------------------------------------
# setting
# ---------------------------------------------------------------------
def samples_file(hyper_paras, info=""):
    # image_size = hyper_paras['image_size']
    M_samples_per_para = hyper_paras['M_samples_per_para']
    sigma = hyper_paras['sigma']
    noise_num = hyper_paras['noise_num']
    data_base = 'data'
    if not os.path.exists(data_base):
        os.makedirs(data_base)
    file = os.path.join(data_base,
                        f'{info}'
                        f'_mu{noise_num}_{sigma}'
                        f'_M{M_samples_per_para}')
    return file

'''
u代表的是函数的中间位置，取值为0-0.5
x代表确定函数以后的取值
'''
def qt(u,x):
    if x<= u:
        q = x/u
    elif x<=2*u:
        q = 2- x/u
    else:
        q = 0
    return q


def get_xs_item(u): # 给定了函数的中心，就能够求得每一个点的取值
    xs_item = np.zeros([26,1])
    for i in range(26):
        xs_item[i] = qt(u,i*0.04) #??
    return xs_item.flatten()


def get_xs(M):
    xs = np.zeros([M,26])
    for i in range(M):
        u = random.uniform(0.1,0.5)
        xs[i] = get_xs_item(u)
    return xs
