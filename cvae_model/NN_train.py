# # ---------------------------------------
# #   load basic packages
# # ---------------------------------------

import numpy as np

np.set_printoptions(suppress=True)  # suppress=True取消科学计数法
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示20行
import matplotlib
# matplotlib.use('Agg')    #服务器使用matplotlib，不会显式图像

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cvxpy as cp
from multiprocessing import Pool

# %
import torch

torch.multiprocessing.freeze_support()
from function import samples_file
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import torch.nn.functional as F
# from utils import samples_file,prior_cov,potential_of_prior_cov,PSF_matrix,blurring_matrix
from visualize import printdict
import pprint
import sys
from collections import OrderedDict
from torch.utils.data import Dataset,DataLoader,random_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 要修改的地方：1. problem setting的复制2.hypers['data_file_prefix'] ，加载数据集的thetas重新说明3.模型的网络结构第一层，以及最后一层；学习率


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
# # ---------------------------------------
# #   problem setting
# # ---------------------------------------
problem_type = 'signal_denoise'
hypers = OrderedDict(
    {"sigma_range": [0.01, 0.03, 0.05], "noise_num": 3,
     "M_samples_per_para": 20000,
     'x_dim': 50,
     'data_dim': 50,
     'sigma_prior': 0.01}
)

H = np.identity(hypers['x_dim'])
hypers['H'] = H

# data_file
data_file_prefix = samples_file(hypers, info=problem_type)
hypers['data_file_prefix'] = data_file_prefix

# # ---------------------------------------
# #   print hyperparameters
# # ---------------------------------------
printdict(hypers)

print('-' * 75)
print(f"number of samples: {hypers['noise_num']}*  {hypers['M_samples_per_para']}  (noise_num*M_samples_per_para)")
print('-' * 75)

hypers['data_file_prefix'] = 'data/signal_denoise_mu3_0.01_0.03_M20000'
print("数据OK!!!" if os.path.exists(hypers['data_file_prefix']+"_0.npz") else '\n\n需要生成训练数据!!!!')

# ==========================================================================
# 加载数据集
#======================================================================
debug=False
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


thetas[:20000] = np.ones([20000,1])*0
thetas[20000:40000] = np.ones([20000,1])
thetas[40000:60000] = np.ones([20000,1])*2
# ========================================================================================
# 模型训练
# ===============================================================================
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, layer_size):
        super(NN, self).__init__() #调用父类的初始化函数
        assert type(layer_size) == list
        self.classMLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.classMLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+2 < len(layer_size):
                self.classMLP.add_module(name="A{:d}".format(i), module=nn.ReLU())


    def forward(self, x):
        x = self.classMLP(x)
        x = torch.nn.functional.softmax(x)
        return x



def normalize(theta, unkown, data):
    theta_n = (theta - thetas.min()) / (thetas.max() - thetas.min())
    unkown[:, 1:-1] = (unkown[:, 1:-1] - xs[:, 1:-1].min(axis=0)) / (xs[:, 1:-1].max(axis=0) - xs[:, 1:-1].min(axis=0))

    data_n = (data - ys.min(axis=0)) / (ys.max(axis=0) - ys.min(axis=0))
    return theta_n, unkown, data_n

ds = DateSet(thetas=thetas, xs=xs, ys=ys)  # ??

train_size = int(len(thetas) * 0.8)
test_size = len(thetas) - train_size

ds_train, ds_test = random_split(ds, [train_size, test_size])
# ？？

batch_size = 64
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

theta_test, unkown_test, data_test = next(iter(test_loader))
num_test = 1000
x_test, y_test = theta_test.to(device)[:num_test, :], torch.cat((unkown_test, data_test), dim=1).to(device)[:num_test, :]

np.random.seed(1)
latentsize = [5]  # 5,10,30,50
for m in latentsize:
    print('@' * 75)
    print(m)
    print('@' * 75)
    for i in np.arange(1e-5, 1e-2, 5e-5):  # 0.0000005,0.000004,0.0000005 1e-8,1e-6,5e-8
        print('@' * 30)
        print(f'latentsize:{m}')
        print(f'lr:{i}')
        print('@' * 30)
        types='unnorm'
        hyper_num = 1  # ??
        classes = 3
        args = {"epochs": 40,
                "batch_size": 64,
                "layer_size": [100,200,100,50,30,classes],
                "print_every": 5000,
                "fig_root": 'figs',
                }
        e = args['epochs']
        s = hypers['noise_num']
        # model_file_name = os.path.join('saved_model', f'0504_class_epoch{epoch}_acc{accuracy}_lr{i}')  # ??
        print(f"total data size:{len(thetas) * args['epochs']}\nbatch_per_epoch:{len(thetas) // args['batch_size']}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



        # # model
        # vae = VAE(encoder_layer_sizes=args['encoder_layer_sizes'],
        #           latent_size=args['latent_size'],
        #           decoder_layer_sizes=args['decoder_layer_sizes'],
        #           conditional=args['conditional'],
        #           num_labels=76 if args['conditional'] else 0).to(device)  # ??更改维数了
        network = NN(layer_size=args['layer_size']).to(device)
        for m in network.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        crossentropyloss = nn.CrossEntropyLoss()

        # optimizer

        optimizer = torch.optim.RMSprop(network.parameters(), lr=i)#SGD(network.parameters(), lr=i,momentum=0.9)
        #
        loss_epoch_mean = []
        loss_epoch_mean_test = []
        for epoch in range(args['epochs']):
            epoch_loss = 0
            epoch_loss_test = 0

            # train
            for iteration, (theta, unkown, data) in enumerate(train_loader):
                if types =='norm':
                    theta, unkown, data = normalize(theta, unkown, data)
                # theta, unkown, data = normalize(theta,unkown,data)
                x, unkown, data = theta.to(device), unkown.to(device), data.to(device)
                # y = torch.cat([unkown, data], dim=1)
                y = data
                recon_x = network(y)  # ??
                loss = crossentropyloss(recon_x,x.flatten().long())

                recon_x_predict = torch.max(recon_x, 1)[1]
                pred_y = recon_x_predict.cpu().data.numpy().squeeze()
                target_y = x.cpu().data.numpy().squeeze()
                accuracy = sum(pred_y == target_y) / x.size(0)
                # print(f"loss:{loss},acc:{accuracy}")


                if iteration % 600 == 0: print(f'loss:{loss.item()},accuracy:{accuracy}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                # if iteration % args['print_every'] == 0 or iteration == len(train_loader)-1:
                if iteration == len(train_loader) - 1:
                    print("==========>Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(epoch + 1,
                                                                                                  args['epochs'],
                                                                                                  iteration,
                                                                                                  len(train_loader) - 1,
                                                                                                  loss.item()))
                    # 利用test_loader部分作predict
                    # c = y_test
                    c = data_test.to(device)
                    x_predict = network(c)
                    loss_test = crossentropyloss(x_predict,x_test.flatten().long())
                    x_predict = torch.max(x_predict,1)[1]
                    pred_y = x_predict.cpu().data.numpy().squeeze()
                    target_y = x_test.cpu().data.numpy().squeeze()
                    accuracy = sum(pred_y == target_y) / y_test.size(0)
                    print(f'=================accuracy:{accuracy}')
                    if accuracy > 0.7:print(f'=================accuracy>0.7=======================:{i}')
                        # torch.save(network, model_file_name + '.pth')
                        # np.save(model_file_name + '_args', args)
                        # np.save(model_file_name + '_hypers', hypers)

            loss_epoch_mean.append(epoch_loss / len(train_loader))
            loss_epoch_mean_test.append(loss_test.item())

        print('*' * 75)

        # if torch.mean(torch.abs(x_predict-x_test))<=0.03:
        print(i)
        plt.plot(loss_epoch_mean)
        plt.plot(loss_epoch_mean_test)
        plt.title(f'learning rate:{i}')
        plt.show()

        # pe = (x_predict.detach().cpu().numpy() - x_test.cpu().numpy()) / x_test.cpu().numpy() * 100
        # _ = pd.DataFrame(pe).boxplot()
        # plt.ylim((-200, 500))
        # plt.ylabel('percentage error')
        # plt.title(f'learning rate:{i}')
        # plt.show()

        # print(1e-3*x_predict.T, 1e-3*x_test.T)
        # print(torch.abs(x_predict - x_test).mean())

        # torch.save(vae,model_file_name+'.pth')
        # np.save(model_file_name+'_args',args)
        # np.save(model_file_name+'_hypers',hypers)