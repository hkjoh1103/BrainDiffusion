# %%
# Library
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torchio as tio
import monai

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from diffusion.data import DataPreprocessing
from diffusion.model import Unet3D, GaussianDiffusion3D
from diffusion.function import *
from diffusion.util import *

# %%
def train(args):
    # define parameters from arguments
    lr = args.lr
    batch_size = args.batch_size
    num_iteration = args.num_iteration
    dropout = args.dropout
    
    time_step = args.time_step
    schedule = args.schedule
    base_channels = args.base_channels
    channel_mults = args.channel_mults
    num_res_blocks = args.num_res_blocks
    time_emb_dim = args.time_emb_dim
    
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir
    
    log_rate = args.log_rate
    save_rate = args.save_rate
    
    if torch.cuda.device_count() >= 2:
        device = torch.device("cuda:0")
        device2 = torch.device("cuda:1")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # make directories
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)    
    
    # data preprocessing
    data = DataPreprocessing(args)
    dataloader = cycle(data.get_dataloader())
    
    x = next(dataloader)
    plt.imsave(os.path.join(result_dir, "sample.png"), x[0,0,16,:,:], cmap='gray')

    # define model & optimizer
    model = Unet3D(
        1,
        base_channels,
        channel_mults,
        num_res_blocks,
        time_emb_dim,
    )
    
    betas = get_schedule(args, s=0.008)
    
    diffusion = GaussianDiffusion3D(
        model,
        (32,32,32),
        1,
        None,
        betas,
    )
    
    opt = optim.Adam(diffusion.parameters(), lr=lr)
    
    # load save files
    ckpt_list = sorted(os.listdir(ckpt_dir))
    # if ckpt_list:
    #     load = torch.load(os.path.join(ckpt_dir, ckpt_list[-1]), map_location=device)
    #     diffusion.load_state_dict(load['net'])
    #     opt.load_state_dict(load['opt'])

    #     for state in opt.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.to(device)
        
    # train loop
    if torch.cuda.device_count() >= 2:
        diffusion = diffusion.to(device2)
    else:
        diffusion = diffusion.to(device)

    train_loss_list = []
    
    for i in range(1, num_iteration+1):
        data = next(dataloader).to(device)
        loss = diffusion(data)
        train_loss_list.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if i % log_rate == 0:
            # calculate running loss
            
            print("Iteration %d / %d | loss %.4f"
                  %(i, num_iteration, np.mean(train_loss_list))
            )
            
        
        if i % save_rate == 0:
            with torch.no_grad():
                # save nii file
                sample = diffusion.sample(1, device=device, y=None, use_ema=False)
                sample = sample[0,0,:,:,:].detach().cpu().numpy()
            
                plt.imsave(os.path.join(result_dir, f"Iteration{i}_sag.png"), sample[16,:,:], cmap='gray')
                plt.imsave(os.path.join(result_dir, f"Iteration{i}_cor.png"), sample[:,16,:], cmap='gray')
                plt.imsave(os.path.join(result_dir, f"Iteration{i}_axi.png"), sample[:,:,16], cmap='gray')
                
                #save one pth file
                save_path = os.path.join(ckpt_dir, 'model_test1.pth')
    
                #save pth file as iteration
                #save_path = os.path.join(ckpt_dir, 'model_iteration%d.pth' %i)
                torch.save({
                    'net': diffusion.state_dict(),
                    'opt': opt.state_dict(),
                    'loss': train_loss_list
                }, save_path)
                print('model saved!')

