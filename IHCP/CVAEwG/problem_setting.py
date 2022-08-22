# # ---------------------------------------
# #   load basic packages
# # ---------------------------------------
import cv2
import numpy as np
np.set_printoptions(suppress=True)          # suppress=True取消科学计数法
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)       # 显示20行
import matplotlib
# matplotlib.use('Agg')    #服务器使用matplotlib，不会显式图像

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cvxpy as cp
from multiprocessing import Pool


#%
import torch
torch.multiprocessing.freeze_support()
from function import samples_file
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchsummary import summary

# from utils import samples_file,prior_cov,potential_of_prior_cov,PSF_matrix,blurring_matrix
from visualize import printdict
import pprint
import sys
from collections import OrderedDict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # ---------------------------------------
# #   problem setting
# # ---------------------------------------
problem_type = 'heat_conduction'
hypers = OrderedDict(
    {"sigma":[0.001,0.005,0.01],"noise_num":3,
     "M_samples_per_para": 50000,
     'x_dim':26,
     'data_dim':50}
)

H = pd.read_csv('H.csv')
hypers['H'] = H
  
W = np.identity(hypers['x_dim'])
hypers['W'] = W
    
# precompute potential function of prior cov
# distM = potential_of_prior_cov(hypers['image_size'],hypers['image_size'])
# hypers['distM'] = distM


# data_file
data_file_prefix = samples_file(hypers,info=problem_type)
hypers['data_file_prefix'] = data_file_prefix

# hypers['data_dim'] = hypers['image_size']**2
# hypers['x_dim'] = hypers['image_size']**2



# # ---------------------------------------
# #   print hyperparameters
# # ---------------------------------------
printdict(hypers)
    
print('-'*75)
print(f"number of samples: {hypers['noise_num']}*  {hypers['M_samples_per_para']}  (noise_num*M_samples_per_para)")
print('-'*75)
