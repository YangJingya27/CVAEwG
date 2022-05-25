
import math
import numpy as np
from numpy import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader,random_split
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import numpy as np
from math import log10, sqrt
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def prior_cov(dist,gamma,d):
    """
    K(i,j) = gamma*exp(-Phi_ij_2/d)

    Parameters
    ----------
    gamma: standard variance
        d: potential_of_prior_cov

    Returns
    -------
        
    """

    K = gamma*np.exp(-dist/d)
    return K

def prior_mean_cov(theta,hypers):
    mu, gamma, d,ratio = theta
    mus = np.array([mu]*hypers['x_dim']).reshape(-1,1)
    T_pr = prior_cov(hypers['distM'], gamma=gamma, d=d)
    return mus,T_pr

# radon metrix
M_PI=math.pi
def MAXX(x,y):
    return x if x > y else y


#% dataset
class DateSet(Dataset):
    def __init__(self, thetas,xs, ys,labels):
        self.thetas = thetas
        self.xs = xs
        self.ys = ys
        self.labels = labels

        

    def __len__(self):
        return len(self.thetas)

    def __getitem__(self, i):
        theta = torch.from_numpy(self.thetas[i]).float()
        img = torch.from_numpy(self.xs[i]).float()
        data = torch.from_numpy(self.ys[i]).float()
        labels = torch.from_numpy(self.labels[i]).float()

        return  theta,img,data,labels






#% others
def gpu2numpy(x):
    return x.cpu().numpy()

def numpy2gpu(x):
    return torch.from_numpy(x).float().cuda()


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


def load_split_data(hypers,debug=False):
    from glob import glob
    data_files = glob(hypers['data_file_prefix']+"*.npz")

    thetas = np.array([])
    xs = np.array([])
    ys = np.array([])
    print(f"load data:",end='')
    for i,data_file in enumerate(data_files):
        print(i,end=' ')
        d = np.load(data_file)
        thetas_part, xs_part, ys_part = d['thetas'].astype(np.float32), d['xs'].astype(np.float32), d['ys'].astype(np.float32)

        thetas = np.vstack([thetas,thetas_part]) if thetas.size else thetas_part
        xs = np.vstack([xs,xs_part]) if xs.size else xs_part
        ys = np.vstack([ys,ys_part]) if ys.size else ys_part
        if debug:break
    print('\n',thetas.shape,xs.shape,ys.shape) 
    return thetas,xs,ys

#---------------------------------------------
#  mcmc sample gibbs
#---------------------------------------------
def sample_cumulated_sum(x_sum_pre,x_square_sum_pre,x_i):
    x_sum = x_sum_pre+x_i
    x_square_sum = x_square_sum_pre+x_i**2
    return x_sum,x_square_sum




