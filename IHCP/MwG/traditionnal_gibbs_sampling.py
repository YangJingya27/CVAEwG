import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from function import p_theta,prepare_,get_theta,get_theta_map,get_sigma_map
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import datetime
import time


m = 26
n = 50
alpha = {'0.001':5e-4,'0.003':1.8e-3,'0.005':1e-2,'0.008':1.5e-2,'0.01':2.5e-2}
beta = {'0.001':1e-6,'0.003':1e-5,'0.005':1e-4,'0.008':64e-6,'0.01':1e-4} # 在原文中提到该值的取值与sigma是平方的关系


H = np.array(pd.read_csv('data/H.csv'))
#Y = np.array(pd.read_csv('y_0.005.csv',header=None)) # 之前模拟产生的数据
Y_true = np.array(pd.read_csv('data/y_true.csv'))
# Y = Y_true + np.random.randn(*Y_true.shape) * 0.01

NMCMC = 10000
burn_in = 5000

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(4)



for sigma_true in [0.001,0.005,0.01]:#
    print(f'sigma_true:{sigma_true}')
    # Y = Y_true + sigma_true * np.random.randn(*Y_true.shape)
    Y = np.array(pd.read_csv(f'data/tradition100000_{sigma_true}.csv', header=None))
    THETA = np.zeros([NMCMC, m])
    SIGMA = np.zeros([NMCMC, 1])

    start = time.time()
    sigma = 0.005 # 随机初始化
    theta0 = get_theta_map(sigma,Y,alpha[f'{sigma_true}']) #由最大后验方法确定初始值
    for num in tqdm(range(NMCMC)):
        for i in range(m): # 首先，按照m个维数来确定出theta的一轮采样
            mu_i, sigma_i = prepare_(i, theta0, H, Y, sigma,alpha[f'{sigma_true}'])
            theta0[i] = get_theta(mu_i, sigma_i)
        sigma = get_sigma_map(theta0, Y, alpha[f'{sigma_true}'], beta[f'{sigma_true}'])  # 由最大后验更新sigma
        # if num >= burn_in:
        THETA[num] = theta0.flatten()
        SIGMA[num] = sigma.flatten()

    theta_mean = np.mean(THETA[burn_in:], axis = 0)
    theta_var = np.var(THETA[burn_in:], axis = 0)
    sigma_mean = np.mean(SIGMA[burn_in:])
    sigma_std = np.std(SIGMA[burn_in:])

    end = time.time()

    np.save(f'data821/tradition_xmean_{sigma_true}.npy', theta_mean)
    np.save(f'data821/tradition_xvar_{sigma_true}.npy', theta_var)
    np.save(f'data821/tradition_theta_{sigma_true}.npy', np.array(SIGMA))
    np.save(f'data821/tradition_x_{sigma_true}.npy', np.array(THETA))
    print('运行时间 : %s 秒' % (end - start))
    print(f'sigma_true:{sigma_true},sigma_mean:{sigma_mean},sigma_std:{sigma_std}')

    x = np.linspace(0, 1, m)
    interval0 = [1 if (i < 0.4) else 0 for i in x]
    interval1 = [1 if (i >= 0.4 and i < 0.8) else 0 for i in x]
    interval2 = [1 if (i >= 0.8) else 0 for i in x]
    y = (2.5 * x) * interval0 + (2 - 2.5 * x) * interval1 + np.array([0]*m) * interval2
    plt.plot(x,y,label = 'ground truth')
    plt.plot(x,theta_mean,label = sigma_true)
    plt.title(f'sigma_true:{sigma_true}')
    plt.legend()
    plt.show()
    plt.savefig(f'figs/IHCP_sigma_true{NMCMC}_{sigma_true}.png')
    print(f'mse:{np.sum((theta_mean -y)**2)/theta_mean.shape[0]},ratio:{np.sum((theta_mean -y)**2)/np.sum(y**2)*100}%')
    #
    # save = pd.DataFrame(theta_mean)
    # save1 = pd.DataFrame(Y)
    # save1.to_csv(f'data/tradition{NMCMC}_{sigma_true}.csv', header=None, index=None)
    # save.to_csv(f'data/y{NMCMC}_{sigma_true}.csv', header=None, index=None)


