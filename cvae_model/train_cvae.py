#!/usr/bin/env python
# coding: utf-8

# In[8]:



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


# In[2]:


problem_type = 'signal_denoise'
hypers = OrderedDict(
    {"sigma_range":[0.01,0.03,0.05],"noise_num":3,
     "M_samples_per_para": 20000,
     'x_dim':50,
     'data_dim':50,
     'sigma_prior':0.1}
)

H = np.identity(hypers['x_dim'])
hypers['H'] = H
# data_file
data_file_prefix = samples_file(hypers,info=problem_type)
hypers['data_file_prefix'] = data_file_prefix

printdict(hypers)
print('-' * 75)
print(f"number of samples: {hypers['noise_num']}*  {hypers['M_samples_per_para']}  (noise_num*M_samples_per_para)")
print('-' * 75)
hypers['data_file_prefix'] = 'data/signal_denoise_mu3_0.01_0.03_M20000' #!!
print("数据OK!!!" if os.path.exists(hypers['data_file_prefix']+"_0.npz") else '\n\n需要生成训练数据!!!!')


# In[3]:


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


# In[4]:


labels = np.zeros([60000,1])  # ！！
labels[:20000] = np.ones([20000,1])*0
labels[20000:40000] = np.ones([20000,1])
labels[40000:60000] = np.ones([20000,1])*2


# In[11]:


from torch.utils.data import Dataset
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


# In[12]:


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
x_test, unkown_test, data_test, label_test = theta_test.to(device)[:num_test, :],                                              unkown_test.to(device)[:num_test, :],                                              data_test.to(device)[:num_test, :],                                              label_test.to(device)[:num_test, :]


# In[14]:


class VAE(nn.Module):

    def __init__(self, feature,classes,encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, num_labels=0):
        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        # assert type(layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.net = NN(feature,classes)
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, unkown, data):
        data_size = data.shape[1]
        data = self.net(data)
        class_data = torch.max(data, 1).indices.float().reshape(-1, 1)
        class_data = class_data @ torch.ones(1,data_size).to(device)
        c = torch.cat([unkown, class_data], dim=1)
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, z, data, class_data[0]

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, unkown, data):
        data_size = data.shape[1]
        data = self.net(data)
        class_data = torch.max(data, 1).indices.float().reshape(-1, 1)
        class_data = class_data @ torch.ones(1,data_size).to(device)
        c = torch.cat([unkown, class_data], dim=1)
        recon_x = self.decoder(z, c)
        return recon_x,data,class_data


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 2 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())

        self.linear_means = nn.Sequential(nn.Linear(layer_sizes[-1], latent_size),nn.BatchNorm1d(latent_size))
        # self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())
                # self.MLP.add_module(name="B{:d}".format(i), module=nn.BatchNorm1d(out_size))
            # else:
            #     self.MLP.add_module(name="Output", module=nn.Softplus())#[batchsize,layer_sizes[-1]]

    # 最后一层不可以是1了...只能是指定的类别
    def forward(self, z, c):
        if self.conditional:
            z = torch.cat((z, c), dim=1)
        x = self.MLP(z)
        out = x
        return out


# class NN(nn.Module):
#     def __init__(self, layer_size):
#         super(NN, self).__init__()  # 调用父类的初始化函数
#         assert type(layer_size) == list
#         self.classMLP = nn.Sequential()
#         for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
#             self.classMLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
#             if i + 2 < len(layer_size):
#                 self.classMLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())

#     def forward(self, data):
#         data = self.classMLP(data)
#         data = torch.nn.functional.softmax(data,dim=1)
#         return data

class Residual_block(nn.Module):
    def __init__(self,in_features,out_features,hidden_features = 100):
        super().__init__()

        self.dense0 = nn.Linear(in_features,hidden_features)
        self.dropout0 = nn.Dropout(p=0.2)

        self.dense1 = nn.Linear(hidden_features,out_features)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dense_skip = nn.Linear(in_features,out_features)

    def forward(self,input):
        l1 = nn.LeakyReLU()(self.dropout0(self.dense0(input)))
        l2 = nn.LeakyReLU()(self.dropout1(self.dense1(l1)))
        skip = nn.LeakyReLU()(self.dense_skip(input))
        output = skip+l2
        return output
    
class NN(nn.Module):
    def __init__(self,feature,classs):
        super().__init__()
        
        self.MLP = nn.Sequential(Residual_block(feature,feature),Residual_block(feature,feature),
                        Residual_block(feature,feature),
                        nn.Linear(feature,classs))
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self,input):
        output = self.MLP(input)
        return output


# In[17]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
lr_list = []
latentsize = [2]
for i in [1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2]:
    print('@' * 30)
    print(f'latentsize:{2}')
    print(f'lr:{i}')
    print('@' * 30)
    
    hyper_num = 1
    classes = 3
    args = {"epochs": 40,
        "batch_size": 64, #!!
        "layer_sizes": [50,100,200,400,200,100,50,20,classes],
        "encoder_layer_sizes": [hyper_num, 200,100,50,30, 10],
        "decoder_layer_sizes": [200,100,50,30, 12, 5,1],
        "latent_size": 2,
        "print_every": 5000,
        "fig_root": 'figs',
        "conditional": True
        }
    
    
    vae = VAE(50,classes,
                  encoder_layer_sizes=args['encoder_layer_sizes'],
                  latent_size=args['latent_size'],
                  decoder_layer_sizes=args['decoder_layer_sizes'],
                  conditional=args['conditional'],
                  num_labels=100 if args['conditional'] else 0).to(device)
    for moduel in vae.modules():
        if isinstance(moduel, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(moduel.weight)
    optimizer = optim.RMSprop([
            {"params": vae.net.parameters(), "lr": 1e-05},
            {"params": vae.encoder.parameters()},
            {"params": vae.decoder.parameters()}],
            lr=i)
    
    loss_epoch_mean = []
    loss_epoch_mean_test = []    
    for epoch in range(args['epochs']):
        epoch_loss = 0
        epoch_loss_test = 0
        for iteration, (theta, unkown, data, label) in enumerate(train_loader):
            # train
            vae.train()
            x, unkown, data, label = theta.to(device), unkown.to(device), data.to(device), label.to(device)
            recon_x, z, recon_data, _ = vae(x, unkown, data)
            true_samples = torch.randn(args['batch_size'], args['latent_size']).to(device)
            if epoch <= 20:
                losses = loss_function(x, recon_x, label, recon_data, true_samples, z, type='cross')
            else:
                losses = loss_function(x, recon_x, label, recon_data, true_samples, z, type='all')
            loss = losses['loss']
            if iteration % 600 == 0: 
                # train metrics
                print(losses['Negative-Loglikelihood'].item(),
                          losses['Maximum_Mean_Discrepancy'].item(),
                          losses['crossentropyloss'].item())
                class_data = torch.max(recon_data, 1)[1]
                pred_y = class_data.cpu().data.numpy().squeeze()
                target_y = label.cpu().data.numpy().squeeze()
                accuracy = sum(pred_y == target_y) / class_data.size(0)
                print(f'********train_accuracy******:{accuracy}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # test
            if iteration == len(train_loader) - 1:
                vae.eval()
                print("==========>Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(epoch + 1,
                                                                                                  args['epochs'],
                                                                                                  iteration,
                                                                                                  len(train_loader) - 1,
                                                                                                  loss.item()))  
                
                z = torch.randn([args['batch_size'], args['latent_size']]).to(device)
                x_predict, recon_data_test, _ = vae.inference(z, unkown_test, data_test)
                class_data = torch.max(recon_data_test, 1)[1]
                pred_y = class_data.cpu().data.numpy().squeeze()
                target_y = label_test.cpu().data.numpy().squeeze()
                accuracy = sum(pred_y == target_y) / class_data.size(0)
                print(f'=================test_accuracy================================:{accuracy}')
                losses_test = loss_function(x_test, x_predict, label_test, recon_data_test, true_samples, z)
                epoch_loss_test += losses_test['loss'].item()
        loss_epoch_mean.append(epoch_loss / len(train_loader))
        loss_epoch_mean_test.append(epoch_loss_test)
    print('*' * 75)
    
    print(x_predict.T, x_test.T)
    print(torch.abs(x_predict - x_test).mean())
    print(recon_x.flatten(), x.flatten())
    
    fig, axs = plt.subplots(2)
    axs[0].plot(loss_epoch_mean[21:])
    axs[0].plot(loss_epoch_mean_test[21:])
    axs[0].set_title(f'learning rate:{i}')

    pe = (x_predict.detach().cpu().numpy() - x_test.cpu().numpy()) / x_test.cpu().numpy() * 100
    axs[1] = pd.DataFrame(pe).boxplot()
    axs[1].set_ylim((-200, 500))
    axs[1].set_ylabel('percentage error')
    # plt.savefig(f'figs/signal_traincvae_class{classes}_learningrate_{i}.png')
    plt.show()
    
    
    if torch.abs(x_predict - x_test).mean() <= 0.013:

        model_file_name = os.path.join('saved_model', f'0524_bn_cvae_class{classes}_lr{i}_accuracy{accuracy}')
        lr_list.append([i,torch.abs(x_predict - x_test).mean(),accuracy])
        torch.save(vae, model_file_name + '.pth')
        np.save(model_file_name + '_args', args)
        np.save(model_file_name + '_hypers', hypers)
 
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
lr_list = []
latentsize = [5]
m = latentsize[0]
for i in [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4]:
    print('@' * 30)
    print(f'latentsize:{m}')
    print(f'lr:{i}')
    print('@' * 30)
    
    hyper_num = 1
    classes = 3
    args = {"epochs": 40,
        "batch_size": 64, #!!
        "layer_sizes": [50,100,200,400,200,100,50,20,classes],
        "encoder_layer_sizes": [hyper_num,200,400,200,100,50,30, 10],
        "decoder_layer_sizes": [200,100,50,30, 12, 5,1],
        "latent_size": m,
        "print_every": 5000,
        "fig_root": 'figs',
        "conditional": True
        }
    
    
    vae = VAE(layer_sizes=args['layer_sizes'],
                  encoder_layer_sizes=args['encoder_layer_sizes'],
                  latent_size=args['latent_size'],
                  decoder_layer_sizes=args['decoder_layer_sizes'],
                  conditional=args['conditional'],
                  num_labels=100 if args['conditional'] else 0).to(device)
    for moduel in vae.modules():
        if isinstance(moduel, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(moduel.weight)
    optimizer = optim.Adam([
            {"params": vae.net.parameters(), "lr": 8e-06},
            {"params": vae.encoder.parameters()},
            {"params": vae.decoder.parameters()}],
            lr=i)
    
    loss_epoch_mean = []
    loss_epoch_mean_test = []    
    for epoch in range(args['epochs']):
        epoch_loss = 0
        epoch_loss_test = 0
        for iteration, (theta, unkown, data, label) in enumerate(train_loader):
            # train
            vae.train()
            x, unkown, data, label = theta.to(device), unkown.to(device), data.to(device), label.to(device)
            recon_x, z, recon_data, _ = vae(x, unkown, data)
            true_samples = torch.randn(args['batch_size'], args['latent_size']).to(device)
            if epoch <= 20:
                losses = loss_function(x, recon_x, label, recon_data, true_samples, z, type='cross')
            else:
                losses = loss_function(x, recon_x, label, recon_data, true_samples, z, type='all')
            loss = losses['loss']
            if iteration % 600 == 0: 
                # train metrics
                print(losses['Negative-Loglikelihood'].item(),
                          losses['Maximum_Mean_Discrepancy'].item(),
                          losses['crossentropyloss'].item())
                class_data = torch.max(recon_data, 1)[1]
                pred_y = class_data.cpu().data.numpy().squeeze()
                target_y = label.cpu().data.numpy().squeeze()
                accuracy = sum(pred_y == target_y) / class_data.size(0)
                print(f'********train_accuracy******:{accuracy}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # test
            if iteration == len(train_loader) - 1:
                vae.eval()
                print("==========>Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(epoch + 1,
                                                                                                  args['epochs'],
                                                                                                  iteration,
                                                                                                  len(train_loader) - 1,
                                                                                                  loss.item()))  
                
                z = torch.randn([args['batch_size'], args['latent_size']]).to(device)
                x_predict, recon_data_test, _ = vae.inference(z, unkown_test, data_test)
                class_data = torch.max(recon_data_test, 1)[1]
                pred_y = class_data.cpu().data.numpy().squeeze()
                target_y = label_test.cpu().data.numpy().squeeze()
                accuracy = sum(pred_y == target_y) / class_data.size(0)
                print(f'=================test_accuracy================================:{accuracy}')
                losses_test = loss_function(x_test, x_predict, label_test, recon_data_test, true_samples, z)
                epoch_loss_test += losses_test['loss'].item()
        loss_epoch_mean.append(epoch_loss / len(train_loader))
        loss_epoch_mean_test.append(epoch_loss_test)
    print('*' * 75)
    
    print(x_predict.T, x_test.T)
    print(torch.abs(x_predict - x_test).mean())
    print(recon_x.flatten(), x.flatten())
    
    fig, axs = plt.subplots(2)
    axs[0].plot(loss_epoch_mean[21:])
    axs[0].plot(loss_epoch_mean_test[21:])
    axs[0].set_title(f'learning rate:{i}')

    pe = (x_predict.detach().cpu().numpy() - x_test.cpu().numpy()) / x_test.cpu().numpy() * 100
    axs[1] = pd.DataFrame(pe).boxplot()
    axs[1].set_ylim((-200, 500))
    axs[1].set_ylabel('percentage error')
    plt.show()
    
    if torch.abs(x_predict - x_test).mean() <= 0.0135:

        # plt.savefig(f'figs/signal_traincvae_class{classes}_learningrate_{i}.png')
        plt.show()
        model_file_name = os.path.join('saved_model', f'0522_BNls{m}_cvae_class{classes}_lr{i}_accuracy{accuracy}')
        lr_list.append([i,torch.abs(x_predict - x_test).mean(),accuracy])
        torch.save(vae, model_file_name + '.pth')
        np.save(model_file_name + '_args', args)
        np.save(model_file_name + '_hypers', hypers)
 


# In[17]:


lr_list


# In[ ]:




