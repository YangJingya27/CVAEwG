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
problem_type = 'signal_denoise'
hypers = OrderedDict(
    {"sigma_range":[0.001, 0.1],"noise_num":5000,
     "M_samples_per_para": 40,
     'x_dim':50,
     'data_dim':50,
     'sigma_prior':0.01}
)


  
H = np.identity(hypers['x_dim']) 
hypers['H'] = H


# data_file
data_file_prefix = samples_file(hypers,info=problem_type)
hypers['data_file_prefix'] = data_file_prefix



# # ---------------------------------------
# #   print hyperparameters
# # ---------------------------------------
printdict(hypers)
    
print('-'*75)
print(f"number of samples: {hypers['noise_num']}*  {hypers['M_samples_per_para']}  (noise_num*M_samples_per_para)")
print('-'*75)
