import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class VAE(nn.Module):
    def __init__(self, layer_sizes, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, num_labels=0):
        super().__init__()
        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.net = NN(layer_sizes)
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)

    def forward(self, x, unkown, data):
        data_size = data.shape[1]
        data = self.net(data)
        class_data = torch.max(data, 1).indices.float().reshape(-1, 1)
        class_data = class_data @ torch.ones(1,data_size)
        c = torch.cat([unkown, class_data], dim=1)
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, z, data, class_data

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, unkown, data):
        data_size = data.shape[1]
        data = self.net(data)
        class_data = torch.max(data, 1).indices.float().reshape(-1, 1)
        class_data = class_data @ torch.ones(1,data_size)
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

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
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
        self.last_par = torch.nn.parameter.Parameter(torch.normal(0, 0.1, size=(4, 1)))

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU())
                self.MLP.add_module(name="B{:d}".format(i), module=nn.BatchNorm1d(out_size))
            # else:
            #     self.MLP.add_module(name="Output", module=nn.Softplus())#[batchsize,layer_sizes[-1]]

    # 最后一层不可以是1了...只能是指定的类别
    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=1)
        x = self.MLP(z)
        out = x
        return out


class NN(nn.Module):
    def __init__(self, layer_size):
        super(NN, self).__init__()  # 调用父类的初始化函数
        assert type(layer_size) == list
        self.classMLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_size[:-1], layer_size[1:])):
            self.classMLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 2 < len(layer_size):
                self.classMLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, data):
        data = self.classMLP(data)
        data = torch.nn.functional.softmax(data,dim=1)
        return data


def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


def loss_function(x, x_reconstructed, label, data_recon, true_samples, z, type='all'):
    nnl = (x_reconstructed - x).pow(2).mean()
    mmd = compute_mmd(true_samples, z)
    crossentropyloss = nn.CrossEntropyLoss()
    cross = crossentropyloss(data_recon, label.flatten().long())
    if type == 'all':
        loss = nnl + mmd + 0.1*cross
    else:
        loss = cross
    return {'loss': loss, 'Negative-Loglikelihood': nnl,
            'Maximum_Mean_Discrepancy': mmd, 'crossentropyloss': cross}