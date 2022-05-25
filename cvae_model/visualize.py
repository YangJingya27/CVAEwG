import numpy as np
from math import log10, sqrt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset,DataLoader,random_split
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
import os
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%b-%d_%H-%M')

def plot_single_image(image):
    """
    Args:
        image:
    Returns:
    """
    if image.ndim==1:
        H = int(image.size ** 0.5)
        W = H
        plt.imshow(image.reshape(H, W), cmap='gray')
    if image.ndim>1:
        m,n = image.shape
        if m==1 or n==1:
            H = int((m*n)**0.5)
            W = H
            plt.imshow(image.reshape(H,W),cmap='gray')
        else:
            plt.imshow(image,cmap='gray')
    plt.colorbar()
    plt.savefig(os.path.join('figs',f'single_image{timestamp}.png'))
    plt.show()

def plot_single_data(y,angleNum=24,cmap='jet'):
    """
    Args:
        image:
    Returns:
    """
    if y.ndim==1:
        H = y.size//angleNum
        W = angleNum
        plt.imshow(y.reshape(H,W,order='F'), cmap=cmap, aspect='auto')
    if y.ndim>1:
        m,n = y.shape
        if m==1 or n==1:
            H = y.size // angleNum
            W = angleNum
            plt.imshow(y.reshape(H,W,order='F'),cmap=cmap,aspect='auto')
        else:
            H, W = y.shape
            plt.imshow(y,cmap=cmap,aspect='auto')
    plt.title("Radon transform\n(Sinogram)")
    plt.xlabel(f"Projection angle (deg):{W}")
    plt.ylabel(f"Projection position (pixels):{H}")
    plt.colorbar()
    plt.savefig(os.path.join('figs',f'single_data{timestamp}.png'))
    plt.show()

def show_images(imgs,cmap='bwr',titles=None, keep_range=True, shape=None, figsize=(8, 8.5)):
    """
    Parameters
    ----------
    imgs   [image1(H,W,C), image2,....., imageN],numpy data
    titles
    keep_range
    shape
    figsize

    Returns
    -------

    """
    combined_data = np.array(imgs)

    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]

    # Get the min and max of all images
    if keep_range:
        _min, _max = np.amin(combined_data), np.amax(combined_data)
    else:
        _min, _max = None, None

    if shape is None:
        shape = (1, len(imgs))

    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True, sharey=True)
    ax = axes.flatten()
    for i, (img, title) in enumerate(zip(imgs, titles)):
        im = ax[i].imshow(img, cmap=cmap,vmin=_min, vmax=_max)
        ax[i].set_title(title)
#         if i==(len(imgs)-1): #最后一个图加colorbar
#             divider = make_axes_locatable(ax[i])
#             cax = divider.append_axes("right", size="10%", pad=0.05)
#             fig.colorbar(im,ax=ax[i],cax=cax)

    plt.savefig(os.path.join('figs',f'images{timestamp}.png'))
    plt.show()

def show_lines(data, titles=None, shape=None, figsize=(8, 8.5)):
    combined_data = np.array(data)
    if titles is None:
        titles = [str(i) for i in range(combined_data.shape[0])]
    if shape is None:
        shape = (1, combined_data.shape[0])
    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=True)
    ax = axes.ravel()
    for i, (d, title) in enumerate(zip(combined_data.T, titles)):
        ax[i].plot(d)
        ax[i].set_title(title)
    plt.savefig(os.path.join('figs',f'lines{timestamp}.png'))
    plt.show()

    
def plot_true_and_predict(res,hypers,title='hyperPara'):
    res.columns=['truth','pred']

    res2 = res.groupby('truth').mean()
    res2['std'] = res.groupby('truth').std()
    res2['cnt'] = res.groupby('truth').count()
    res2.columns = ['mean','std','cnt']
    res2 = res2.reset_index()
    res2['pe'] = (res2['truth']-res2['mean']).abs()/res2['truth']*100


    print(res2[res2.cnt>20].sort_values(by='pe').tail())
    print(res2[res2.cnt>20].sort_values(by='pe').head())

    fig, ax = plt.subplots()
    plt.errorbar(res2['truth'],res2['mean'], yerr=res2['std'])
    plt.xlabel('truth')
    plt.ylabel('predict')
    lims = [hypers['d_range'][0],hypers['d_range'][1]]
    plt.xlim(lims)
    plt.ylim(lims)
    ax.plot([0,1],[0,1], transform=ax.transAxes)
    plt.grid()
    plt.axis('square')
    plt.title(title)
    
def printdict(hypers,shape_of_key=['H','W']):
    for key, value in hypers.items():
        if key in shape_of_key:
            print(f"{key:>5}: {value.shape}")
        else:
            print(f"{key:>5}: {value}")
