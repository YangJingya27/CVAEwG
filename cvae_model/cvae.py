from torch.utils.data import DataLoader,random_split
import os
import cv2
import matplotlib
# matplotlib.use('Agg')    #服务器使用matplotlib，不会显式图像
import matplotlib.pyplot as plt
import os
import torch
torch.multiprocessing.freeze_support()
from function import samples_file,DateSet
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from visualize import printdict
from collections import OrderedDict
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from cvae_models import VAE,loss_function
import numpy as np
np.set_printoptions(suppress=True)  # suppress=True取消科学计数法
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示20行
# # ---------------------------------------
# #   problem setting
# # ---------------------------------------
problem_type = 'signal_denoise'
hypers = OrderedDict(
    {"sigma_range":[0.01,0.03,0.05],"noise_num":3,
     "M_samples_per_para": 20000,
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
print('-' * 75)
print(f"number of samples: {hypers['noise_num']}*  {hypers['M_samples_per_para']}  (noise_num*M_samples_per_para)")
print('-' * 75)
hypers['data_file_prefix'] = 'data\signal_denoise_mu3_0.01_0.03_M20000' #!!
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

labels = np.zeros([60000,1])  # ！！
labels[:20000] = np.ones([20000,1])*0
labels[20000:40000] = np.ones([20000,1])
labels[40000:60000] = np.ones([20000,1])*2

def normalize(theta, unkown, data):
    theta_n = (theta - thetas.min()) / (thetas.max() - thetas.min())
    unkown[:, 1:-1] = (unkown[:, 1:-1] - xs[:, 1:-1].min(axis=0)) / (xs[:, 1:-1].max(axis=0) - xs[:, 1:-1].min(axis=0))
    data_n = (data - ys.min(axis=0)) / (ys.max(axis=0) - ys.min(axis=0))
    return theta_n, unkown, data_n


types = 'notnorm'
batch_size = 64
ratio = 0.8
ds = DateSet(thetas=thetas, xs=xs, ys=ys, labels=labels)  # ??
train_size = int(len(thetas) * ratio)
test_size = len(thetas) - train_size
ds_train, ds_test = random_split(ds, [train_size, test_size])
train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

theta_test, unkown_test, data_test, label_test = next(iter(test_loader))
num_test = 1000
x_test, unkown_test, data_test, label_test = theta_test.to(device)[:num_test, :], \
                                             unkown_test.to(device)[:num_test, :], \
                                             data_test.to(device)[:num_test, :], \
                                             label_test.to(device)[:num_test, :]

latentsize = [5]  # 5,10,30,50
for m in latentsize:
    print('@' * 75)
    print(m)
    print('@' * 75)
    # for i in np.arange(1e-6, 1e-2, 5e-6):  # 0.0000005,0.000004,0.0000005 1e-8,1e-6,5e-8
    for i in np.arange(5e-7, 5e-6, 2e-7): #1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2
        print('@' * 30)
        print(f'latentsize:{m}')
        print(f'lr:{i}')
        print('@' * 30)

        types = 'notnorm'
        hyper_num = 1  # ??
        classes = 3  # ！！
        args = {"epochs": 50,
                "batch_size": 64, #!!
                "layer_sizes": [50, 100, 200, 100, 50, 30, classes],
                "encoder_layer_sizes": [hyper_num, 30, 12, 5],
                "decoder_layer_sizes": [40, 20, 10, 1],
                "latent_size": m,
                "print_every": 5000,
                "fig_root": 'figs',
                "conditional": True
                }
        e = args['epochs']
        s = hypers['noise_num']
        # model
        vae = VAE(layer_sizes=args['layer_sizes'],
                  encoder_layer_sizes=args['encoder_layer_sizes'],
                  latent_size=args['latent_size'],
                  decoder_layer_sizes=args['decoder_layer_sizes'],
                  conditional=args['conditional'],
                  num_labels=100 if args['conditional'] else 0).to(device)  # ??更改维数了
        for moduel in vae.modules():
            if isinstance(moduel, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(moduel.weight)

        # optimizer
        optimizer = optim.NAdam([
            {"params": vae.net.parameters(), "lr":1.19e-05 }, # 3.7e-06
            {"params": vae.encoder.parameters()},
            {"params": vae.decoder.parameters()}],
            lr=i)  # torch.optim.Adam(vae.parameters(), lr=lr)

        loss_epoch_mean = []
        loss_epoch_mean_test = []
        for epoch in range(args['epochs']):
            epoch_loss = 0
            epoch_loss_test = 0
            # train
            for iteration, (theta, unkown, data, label) in enumerate(train_loader):
                if types == 'norm': theta, unkown, data = normalize(theta, unkown, data)
                x, unkown, data, label = theta.to(device), unkown.to(device), data.to(device), label.to(device)
                recon_x, z, recon_data, _ = vae(x, unkown, data)  # ?
                true_samples = torch.randn(args['batch_size'], args['latent_size'])
                true_samples = true_samples.to(device)

                if epoch <= 30:
                    losses = loss_function(x, recon_x, label, recon_data, true_samples, z, type='cross')
                else:
                    losses = loss_function(x, recon_x, label, recon_data, true_samples, z, type='all')
                loss = losses['loss']

                if iteration % 600 == 0:
                    print(losses['Negative-Loglikelihood'].item(),
                          losses['Maximum_Mean_Discrepancy'].item(),
                          losses['crossentropyloss'].item())

                    class_data = torch.max(recon_data, 1)[1]
                    pred_y = class_data.cpu().data.numpy().squeeze()
                    target_y = label.cpu().data.numpy().squeeze()
                    accuracy = sum(pred_y == target_y) / class_data.size(0)
                    print(f'=================train_accuracy:{accuracy}')

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
                    z = torch.randn([args['batch_size'], args['latent_size']]).to(device)
                    x_predict, recon_data_test, _ = vae.inference(z, unkown_test, data_test)

                    class_data = torch.max(recon_data_test, 1)[1]
                    pred_y = class_data.cpu().data.numpy().squeeze()
                    target_y = label_test.cpu().data.numpy().squeeze()
                    accuracy = sum(pred_y == target_y) / class_data.size(0)
                    print(f'=================test_accuracy================================:{accuracy}')

                    # if accuracy >= 0.65:
                    #     torch.save(vae, model_file_name + '.pth')
                    #     np.save(model_file_name + '_args', args)
                    #     np.save(model_file_name + '_hypers', hypers)

                    losses_test = loss_function(x_test, x_predict, label_test, recon_data_test, true_samples, z)
                    loss_test = losses_test['loss']
                    epoch_loss_test += loss_test.item()
            loss_epoch_mean.append(epoch_loss / len(train_loader))
            loss_epoch_mean_test.append(epoch_loss_test)
        print('*' * 75)

        # if torch.mean(torch.abs(x_predict-x_test))<=0.03:
        print(i)

        fig, axs = plt.subplots(2)
        axs[0].plot(loss_epoch_mean)
        axs[0].plot(loss_epoch_mean_test)
        axs[0].set_title(f'learning rate:{i}')

        pe = (x_predict.detach().cpu().numpy() - x_test.cpu().numpy()) / x_test.cpu().numpy() * 100
        axs[1] = pd.DataFrame(pe).boxplot()
        axs[1].set_ylim((-200, 500))
        axs[1].set_ylabel('percentage error')
        plt.savefig(f'figs/signal_traincvae_class{classes}_learningrate_{i}.png')
        plt.show()

        print(x_predict.T, x_test.T)
        print(torch.abs(x_predict - x_test).mean())
        print(recon_x.flatten(), x.flatten())
        model_file_name = os.path.join('../../train_cvae分开/cvae_model/saved_model', f'0519_cvae_class{classes}_lr{i}_accuracy{accuracy}')
        if torch.abs(x_predict - x_test).mean() <= 0.0025:
            torch.save(vae, model_file_name + '.pth')
            np.save(model_file_name + '_args', args)
            np.save(model_file_name + '_hypers', hypers)