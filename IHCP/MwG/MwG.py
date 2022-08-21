import numpy as np
import torch
from utils import sample_cumulated_sum
import pandas as pd
import os
from tqdm import tqdm
from function import get_x_ml, W_matrx_1,get_sigma_map,x_post_sample_modify,log_posterior_sigma
import matplotlib.pyplot as plt
import time

model_file_name = '0506_ls5_cvae_lr9.599999999999998e-05' #??

args = np.load(os.path.join('saved_model',model_file_name)+"_args.npy",allow_pickle=True).item()
hypers = np.load(os.path.join('saved_model',model_file_name)+"_hypers.npy",allow_pickle=True).item()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(4)

def x_post_sample_modify(W,H,m,sigma,y,alpha): # 检查过，没有问题
    sigma = sigma + 1e-8
    lamda = 2*alpha/(sigma**2)
#     T_pr_inv = np.linalg.inv(W/lamda)
    T_pr_inv = W * lamda
    post_cov = np.linalg.inv((1/sigma**2)* H.T@H + T_pr_inv)
    post_mean = post_cov@(H.T@y* (1/sigma**2))

    post_cov_L = np.linalg.cholesky(post_cov)
    x_post_sample = post_mean+post_cov_L@np.random.randn(m,1)
    return x_post_sample,post_mean,post_cov

def log_posterior_sigma(theta,Y,hypers,sigma,beta,alpha):
    H = np.array(hypers['H'])
    W = W_matrx_1()

    m = hypers['x_dim']

    C = (H @ theta - Y).T @ (H @ theta - Y) + 2 * alpha * theta.T @ W @ theta
    log_pos = (n + m) * np.log(sigma) + C / (2 * sigma ** 2) + sigma ** 2 / beta
    return -log_pos

total = 10000
for sigma_noise in [0.001,0.005,0.01]:
    for N in [10000,1000,100]: #[100000,10000,1000,100]
        use_sample_size,M = N/2, int(total/N)
        y_true = np.array(pd.read_csv('data/y_true.csv'))
        x_true = np.array(pd.read_csv('data/qt_true.csv',header = None))
        y = np.array(pd.read_csv(f'data/tradition100000_{sigma_noise}.csv',header=None))
        # y = y_true + 0.01 *np.random.randn(*y_true.shape)
        # print(np.sum(np.abs(y-y_true))/np.sum(np.abs(y_true)))

        H = hypers['H']
        H_arr = np.array(H)
        W = W_matrx_1()
        # alpha = hypers['alpha']
        alpha = {'0.001':5e-4,'0.003':1.8e-3,'0.005':1e-2,'0.008':1.5e-2,'0.01':2.5e-2}
        beta = {'0.001':1e-6,'0.003':1e-5,'0.005':1e-4,'0.008':6.4e-5,'0.01':1e-4} # 与sigma是平方的关系
        # hypers['beta'] = 1e-6
        # beta = hypers['beta']
        m = hypers['x_dim']
        n = hypers['data_dim']


        #x_traditional = np.array(pd.read_csv('tradition0.01_1.csv',header=None))

        theta_truth = np.zeros(1)#??
        # np.random.seed(0)
        x_sum,x_square_sum = np.zeros_like(x_true),np.zeros_like(x_true)
        theta_sum,theta_square_sum = np.zeros_like(theta_truth),np.zeros_like(theta_truth)

        # x_0 = x_true
        x_0 = get_x_ml(y,hypers) # MLE
        # sigma0 = get_sigma_map(x_0, y)
        sigma0 = 0.005 # random
        var0 = 0.005**2
        print('-----',sigma_noise)
        step_size = 5e-4
        accept = 0 #np.zeros([N,1]) #
        trace_plot = np.zeros([N,M,1])
        start = time.time()
        THETA = []
        X = []
        for i in tqdm(range(N)):
            for j in range(M):
                sigma_star = sigma0 + step_size * np.random.normal()
                prob = np.exp(log_posterior_sigma(x_0, y, hypers, sigma_star, beta[str(sigma_noise)], alpha[str(sigma_noise)])
                    - log_posterior_sigma(x_0, y, hypers, sigma0, beta[str(sigma_noise)], alpha[str(sigma_noise)]))
                if np.random.rand() < prob:
                    sigma0 = sigma_star
                    accept = accept + 1
                else:
                    sigma0 = sigma0
                trace_plot[i, j] = sigma0
            sigma0 = np.mean(trace_plot[i])

            # sample x
            # x_0, _, _ = x_post_sample_modify(W, H_arr, m, np.sqrt(var0), y, alpha[str(sigma_noise)])  # ??
            x_0, _, _ = x_post_sample_modify(W, H_arr, m, sigma0, y, alpha[str(sigma_noise)])

            if i>(N-use_sample_size-1):
                # theta_sum,theta_square_sum = sample_cumulated_sum(theta_sum,theta_square_sum,np.sqrt(var0))
                theta_sum, theta_square_sum = sample_cumulated_sum(theta_sum, theta_square_sum, sigma0)
                x_sum,x_square_sum = sample_cumulated_sum(x_sum,x_square_sum,x_0)
            THETA.append(sigma0)
            X.append(x_0)

        x_mean = x_sum/use_sample_size
        x_var = x_square_sum/use_sample_size-x_mean**2
        theta_mean = theta_sum/use_sample_size
        theta_var = theta_square_sum/use_sample_size-theta_mean**2
        end = time.time()

        x = np.linspace(0, 1, m)
        interval0 = [1 if (i < 0.4) else 0 for i in x]
        interval1 = [1 if (i >= 0.4 and i < 0.8) else 0 for i in x]
        interval2 = [1 if (i >= 0.8) else 0 for i in x]
        y = (2.5 * x) * interval0 + (2 - 2.5 * x) * interval1 + np.array([0] * m) * interval2
        rmse = np.sqrt(np.sum((x_mean.flatten()-y)**2)/x.shape[0])

        np.save(f'data821/mwg_xmean_{sigma_noise}_N{N}_M{M}_{accept / total}.npy',x_mean)
        np.save(f'data821/mwg_xvar_{sigma_noise}_N{N}_M{M}_{accept / total}.npy',x_var)
        np.save(f'data821/mwg_theta_{sigma_noise}_N{N}_M{M}_{accept / total}.npy', np.array(THETA))
        np.save(f'data821/mwg_x_{sigma_noise}_N{N}_M{M}_{accept / total}.npy', np.array(X))


        print(N,M,accept / total)
        print(f'rmse:{rmse}')
        print(f'sigma_true:{sigma_noise},sigma_mean:{theta_mean},sigma_var:{np.sqrt(theta_var)}')
        print('运行时间 : %s 秒' % (end - start))

    # x = np.linspace(0, 1, m)
    # interval0 = [1 if (i < 0.4) else 0 for i in x]
    # interval1 = [1 if (i >= 0.4 and i < 0.8) else 0 for i in x]
    # interval2 = [1 if (i >= 0.8) else 0 for i in x]
    # y = (2.5 * x) * interval0 + (2 - 2.5 * x) * interval1 + np.array([0]*m) * interval2

    # print(np.sqrt(np.sum((x_mean.flatten()-y)**2)/x.shape[0]))######################3！！！！！！！！！！！！！！！1
    # print(accept)
    # plt.plot(x, y)
    # plt.plot(x, x_mean, linestyle='--')
    # plt.legend(('True heat flux', 'Baseline'))
    # plt.show()

    # x = np.linspace(0, 10000, N)
    # plt.plot(x, trace_plot)
    # plt.show()
