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

# image debluring inverse problem
def PSF_matrix(image_size,t=0.002,r=5):
    """
    from:An approximate empirical Bayesian method
for large-scale linear-Gaussian inverse
problems

    Parametersl
    ----------
    image_size
            t
            r point spread function离散化后得到的离散矩阵半径

    Returns
    -------

    """
    t1_ind=t2_ind=image_size//2
    xs = np.linspace(0, 1, image_size + 1)
    delta_x = (xs[1] - xs[0]) / 2
    xs = xs[:-1] + delta_x

    ys = np.linspace(0, 1, image_size + 1)
    ys = ys[:-1] + delta_x

    tao1 = xs[t1_ind-r:t1_ind+r+1]
    tao2 = ys[t2_ind-r:t2_ind+r+1]

    PSF = np.zeros((2*r+1,2*r+1))
    for i in range(2*r+1):
        for j in range(2*r+1):
            temp = (tao1[r]-tao1[i])**2+(tao2[r]-tao2[j])**2
            PSF[i,j] = np.exp(-temp/t)

    # plt.imshow(PSF)
    # plt.show()
    return PSF

def blurring_matrix(r,t,hypers):
    """

    Parameters
    ----------
            r point spread function离散化后得到的离散矩阵半径
            t 参数
    hypers

    Returns
    -------

    """
    row = col = hypers['image_size']
    PSF = PSF_matrix(row, t=t, r=r)
    nx = row*col
    G = np.zeros((nx, nx))
    for i in range(row):
        for j in range(col):
            # y_ij = (k*x)_ij
            for k in range(r, -r - 1, -1):
                for l in range(r, -r - 1, -1):
                    cond1 = 0 <= (i - k) <= (row - 1)
                    cond2 = 0 <= (j - l) <= (col - 1)
                    if cond1 & cond2:
                        # print(i,j,k,l)
                        G[i * col + j, (i - k) * col + j - l] = PSF[r + k, r + l]
    return G

# isotropic Gassian covriance
def potential_of_prior_cov(height,width):
    """
    0<=x<=1
    0<=y<=1
    
    Phi_ij = ||(x_i,y_i)-(x_j,y_j)||_2
    """
    xs = np.linspace(0,1,width+1)
    delta_x = (xs[1]-xs[0])/2
    xs = xs[:-1]+delta_x

    ys = np.linspace(0,1,height+1)
    ys = ys[:-1]+delta_x

    [X,Y] = np.meshgrid(xs,ys)
    XYpoint = np.c_[X.flatten(),Y.flatten()]

    n = height*width
    dist_upper = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            dist_upper[i,j] = np.linalg.norm(XYpoint[i,:]-XYpoint[j,:])
    distMatrix = dist_upper+dist_upper.T   
    return distMatrix


    
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

def radon_matrix(nt,nx,ny):
    """
    nt      angle num
    nx,ny = width, height
    """
    xOrigin = int(MAXX(0, math.floor(nx / 2)))
    yOrigin = int(MAXX(0, math.floor(ny / 2)))
    Dr = 1
    Dx = 1
    rsize=math.floor(math.sqrt(float(nx*nx+ny*ny)*Dx)/(2*Dr))+1    # from zhang xiaoqun
    # rsize = int(math.sqrt(2)*MAXX(nx,ny)/2)
    nr=2*rsize+1
    xTable = np.zeros((1,nx))
    yTable = np.zeros((1,ny))
    yTable[0,0] = (-yOrigin - 0.5) * Dx
    xTable[0,0] = (-xOrigin - 0.5) * Dx
    for i in range(1,ny):
        yTable[0,i] = yTable[0,i-1] + Dx
    for ii in range(1,nx):
        xTable[0,ii]=xTable[0,ii-1] + Dx
    Dtheta = M_PI / nt
    percent_sparse = 2/ float(nr)
    nzmax = int(math.ceil(float(nr * nt * nx * ny * percent_sparse)))
    # nr=len(rho)
    # nt=len(theta)
    A= np.zeros((nr * nt,nx * ny))
    weight = np.zeros((1,nzmax))
    irs = np.zeros((1,nzmax))
    jcs =np.zeros((1,A.shape[1]+1))
    k=0
    for m in range(ny):
        for n in range(nx):
            jcs[0,m*nx+n]=k
            for j in range(nt):
                angle=j*Dtheta
                cosine=math.cos(angle)
                sine=math.sin(angle)
                xCos=yTable[0,m]*cosine+rsize*Dr
                ySin=xTable[0,n]*sine
                rldx=(xCos+ySin)/Dr
                rLow=math.floor(rldx)
                pixelLow=1-rldx+rLow
                if 0 <= rLow < (nr - 1):
                    irs[0,k]=nr*j+rLow #irs为元素储存的行号
                    weight[0,k]=pixelLow
                    k=k+1
                    irs[0,k]=nr*j+rLow+1
                    weight[0,k]=1-pixelLow
                    k=k+1
        jcs[0,nx * ny] = k
    for col in range(nx*ny):
        for row in range(2*nt):
            A[int(irs[0,col*2*nt+row]),col]=weight[0,col*2*nt+row]
    return np.flipud(A)


#%
def gradient_operater_of_sub_img(H=32,W=32):
    """
    近似的梯度算子,误差分析见https://github.com/zhouqp631/hyperpara_optim_dnn/blob/main/gradient_operator.py
    Parameters
    ----------
    H
    W

    Returns
    -------
    D_approximate
    """
    x_dim = H * W
    index = np.arange(x_dim).reshape(H, W)
    D = np.zeros((x_dim,x_dim))

    select1 = np.linspace(0,H-2,H//2).astype(np.int8)
    select2 = np.linspace(1,W-1,W//2).astype(np.int8)

    start = 0
    for i in select1:
        for j in select1:
            D[start,index[i,j]] = -1
            D[start,index[i,j+1]] = 1
            start += 1
            D[start,index[i,j]] = -1
            D[start,index[i+1,j]] = 1
            start += 1

    for i in select2:
        for j in select2:
            D[start,index[i,j]] = 1
            D[start,index[i-1,j]] = -1
            start += 1
            D[start,index[i,j]] = 1
            D[start,index[i,j-1]] = -1
            start += 1
    return D

#% metrics
def PSNR(ground_truth, predict):
    """
    计算单个图片的PSNR
    """
    ground_truth = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    predict = (predict - predict.min()) / (predict.max() - predict.min())
    mse = np.mean((ground_truth - predict) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return -1
    max_pixel = 1.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return np.round(psnr,2)


#% loss
def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))

#% dataset
class DateSet(Dataset):
    def __init__(self, thetas,xs, ys):
        self.thetas = thetas
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.thetas)

    def __getitem__(self, i):
        theta = torch.from_numpy(self.thetas[i]).float()
        img = torch.from_numpy(self.xs[i]).float()
        data = torch.from_numpy(self.ys[i]).float()
        return  theta,img,data



#% others
def gpu2numpy(x):
    return x.cpu().numpy()

def numpy2gpu(x):
    return torch.from_numpy(x).float().cuda()


def samples_file(hyper_paras,info=""):
    image_size = hyper_paras['image_size']
    M_samples_per_para = hyper_paras['M_samples_per_para']
    mu_range = hyper_paras['mu_range']
    gamma_range = hyper_paras['gamma_range']
    d_range = hyper_paras['d_range']
    
    mu_nums = hyper_paras['mu_nums']
    gamma_nums = hyper_paras['gamma_nums']
    d_nums = hyper_paras['d_nums']
    data_base = 'data'
    if not os.path.exists(data_base):
        os.makedirs(data_base)
    file = os.path.join(data_base,
                f'{info}_size{image_size}'
                f'_mu{mu_nums}_{mu_range[0]}_{mu_range[1]}'
                f'_gamma{gamma_nums}_{gamma_range[0]}_{gamma_range[1]}'
                f'_d{d_nums}_{np.round(d_range[0],4)}_{np.round(d_range[1],4)}'
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


#---------------------------------------------
#  image inverse problem part
#---------------------------------------------
# 与gen_samples_parallel.py中gen_one_para_samples类似，只不多是
# 【指定超参数】产生测试数据

def gen_tests_of_denoise(theta, hypers, M=1):
    """
    x = N(u,C)
    y = Gx+\eta

        Parameters
        ----------
             theta: [mu,gamma,d]
            hypers:

        Returns
            thetas, xs, ys:  [M,dim]
        -------

    """
    x_dim = hypers['x_dim']
    mu, gamma, d = theta

    C = prior_cov(hypers['distM'], gamma=gamma, d=d)
    xs = np.random.multivariate_normal([mu] * x_dim, C, M)  # [M,dim]

    sigma = hypers['noise_signal_ratio'] * xs.max()
    b = np.random.randn(*xs.shape)*sigma
    ys = xs + b

    T_pr = C
    T_obs_inv = np.diag([1/sigma**2]*hypers['x_dim'])

    mus = np.array([mu]*hypers['x_dim']).reshape(-1,1)
    return  xs.reshape(-1,1), ys.reshape(-1,1),T_pr,T_obs_inv,mus

def gen_tests_of_deblur(theta, hypers, M=1,seed=3):
    """
    x = N(u,C)
    y = Gx+\eta

        Parameters
        ----------
             theta: [mu,gamma,d]
            hypers:

        Returns
            thetas, xs, ys:  [M,dim]
        -------

    """
    G = hypers['G']
    x_dim = hypers['x_dim']
    mu, gamma, d,ratio = theta

    C = prior_cov(hypers['distM'], gamma=gamma, d=d)
    random.seed(seed)
    xs = np.random.multivariate_normal([mu] * x_dim, C, M)  # [M,dim]
    # sigma = hypers['noise_signal_ratio'] * xs.max()
    sigma = ratio * xs.max()
    random.seed(seed)
    b = np.random.randn(*xs.shape)*sigma
    ys = xs@G.T + b  # broadcasting    (M,x_dim)+(M,1)

    T_pr = C
    T_obs_inv = np.diag([1/sigma**2]*hypers['x_dim'])

    mus = np.array([mu]*hypers['x_dim']).reshape(-1,1)
    return  xs.reshape(-1,1), ys.reshape(-1,1),T_pr,T_obs_inv,mus


def x_post_sample_modify(T_pr,H,GTy,mus,G=None):
    if G is None:
        G = np.eye(len(mus))
        
    T_pr_inv = np.linalg.inv(T_pr+1e-8*np.eye(T_pr.shape[0]))
    post_cov = np.linalg.inv(H+T_pr_inv)
    post_mean = post_cov@(GTy+T_pr_inv@mus)
    
    post_cov_L = np.linalg.cholesky(post_cov)
    x_post_sample = post_mean+post_cov_L@np.random.randn(len(mus),1)
    return x_post_sample,post_mean,post_cov


# x_truth, y,T_pr,T_obs_inv,mus = gen_tests_of_denoise([2,0.1,0.2],hypers,M=1)
# x_post = x_post_sample(T_pr,T_obs_inv,mus,y)


#---------------------------------------------
# LGC model part
#---------------------------------------------
# 与gen_samples_parallel.py中gen_one_para_samples类似，只不多是
# 【指定超参数】产生测试数据
def gen_test_samples(theta, hypers,M=1):
    """
    x = N(u,C)
    y = Poisson(x*noise_level)/nosie_level

        Parameters
        ----------
             theta: [mu,gamma,d]
            hypers:

        Returns
            thetas, xs, ys:  [M,dim]
        -------

    """
    mu, gamma, d = theta
    
    x_dim = hypers['x_dim']
    noise_level = hypers['noise_level']
  
    C = prior_cov(hypers['distM'], gamma=gamma, d=d)
    xs = np.random.multivariate_normal([mu] * x_dim, C, M)  # [M,dim]
    ys = np.random.poisson(np.exp(xs) * noise_level) / noise_level
    thetas = np.array([theta] * M)
    return [mu]*x_dim,C, xs, ys



#-------------------------------------------
# model predict part
#-------------------------------------------
def predict(net,test_loader,device,index_of_predict):
    """
    预测结果
    """
    thetas_test_all = np.array([])
    thetas_test_pred_all = np.array([])

    for thetas_test, xs_test, ys_test in test_loader:
        thetas_test_pred_gpu = net(xs_test.to(device),ys_test.to(device))
        thetas_test_pred_cpu = thetas_test_pred_gpu.detach().cpu().numpy().reshape(-1,len(index_of_predict))

        thetas_test_all = np.vstack([thetas_test_all,thetas_test[:,index_of_predict]])  if thetas_test_all.size else thetas_test[:,index_of_predict]
        thetas_test_pred_all = np.vstack([thetas_test_pred_all,thetas_test_pred_cpu])   if thetas_test_pred_all.size else thetas_test_pred_cpu

    print(thetas_test_all.shape,thetas_test_pred_all.shape)
    result = pd.DataFrame(np.c_[thetas_test_all,thetas_test_pred_all])
    return result