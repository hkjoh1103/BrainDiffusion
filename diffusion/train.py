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
    name = args.name
    
    lr = args.lr
    batch_size = args.batch_size
    num_iteration = args.num_iteration
    dropout = args.dropout
    
    time_step = args.time_step
    schedule = args.schedule
    image_size = args.image_size
    base_channels = args.base_channels
    channel_mults = args.channel_mults
    num_res_blocks = args.num_res_blocks
    time_emb_dim = args.time_emb_dim
    
    data_dir = args.data_dir
    base_dir = f'./{name}'
    ckpt_dir = os.path.join(base_dir, 'checkpoint')
    log_dir = os.path.join(base_dir, 'log')
    result_dir = os.path.join(base_dir, 'result')
    
    log_rate = args.log_rate
    save_rate = args.save_rate
    
    print('device count: %d' %(torch.cuda.device_count()))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print('device : %s' %device)
    
    # make directories
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)    
    
    # data preprocessing
    data = DataPreprocessing(args)
    dataloader = data.get_dataloader()
 
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
        (image_size, image_size, image_size),
        1,
        None,
        betas,
    )
    
    opt = optim.Adam(diffusion.parameters(), lr=lr)
    
    # check sample image and noise sequence
    x = next(dataloader)
    sample_data = x['flair']['data'][0, 0, :, :, :]
    
    plt.imsave(os.path.join(result_dir, "sample.png"), sample_data[:, :, image_size//2], cmap='gray')

    sample_sequence = [sample_data]
    
    for t in range(time_step):
        noise = torch.randn_like(sample_data)
        
        c1 = np.sqrt(1 - betas[t])
        c1 = torch.tensor(c1, dtype=torch.float32)
        c2 = np.sqrt(betas[t])
        c2 = torch.tensor(c2, dtype=torch.float32)
        
        sample_data = c1 * sample_data + c2 * noise
        
        sample_sequence.append(sample_data)
        
    plt.figure(figsize=(20,8))
    for i in range(1, 11):
        plt.subplot(1,10,i)
        plt.imshow(sample_sequence[(i-1)*(time_step)//10][:, :, image_size//2], cmap='gray')
    plt.savefig(os.path.join(result_dir, "sample_noise_sequence.png"))
    
    # load save files
    # ckpt_list = sorted(os.listdir(ckpt_dir))
    # if ckpt_list:
    #     load = torch.load(os.path.join(ckpt_dir, ckpt_list[-1]), map_location=device)
    #     diffusion.load_state_dict(load['net'])
    #     opt.load_state_dict(load['opt'])

    #     for state in opt.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.to(device)
        

        
    # train loop
    diffusion = diffusion.cuda()
    train_loss_list = []
    
    for i in range(1, num_iteration+1):
        data = next(dataloader)['flair']['data'].cuda()
        
        loss = diffusion(data)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss_list.append(loss.item())

        
        if i % log_rate == 0:
            # calculate running loss
            
            print("Iteration %d / %d | loss %.4f"
                  %(i, num_iteration, np.mean(train_loss_list[-log_rate:]))
            )
            
        
    #     if i % save_rate == 0:
    #         with torch.no_grad():
    #             # save nii file
    #             sample = diffusion.sample(batch_size, device=device, y=None, use_ema=False)
    #             sample = sample[0,0,:,:,:].detach().cpu().numpy()
    #             _m = image_size // 2
            
    #             plt.imsave(os.path.join(result_dir, f"Iteration{i}_sag.png"), sample[_m,:,:], cmap='gray')
    #             plt.imsave(os.path.join(result_dir, f"Iteration{i}_cor.png"), sample[:,_m,:], cmap='gray')
    #             plt.imsave(os.path.join(result_dir, f"Iteration{i}_axi.png"), sample[:,:,_m], cmap='gray')
                
    #             #save one pth file
    #             save_path = os.path.join(ckpt_dir, 'model_test1.pth')
    
    #             #save pth file as iteration
    #             #save_path = os.path.join(ckpt_dir, 'model_iteration%d.pth' %i)
    #             torch.save({
    #                 'net': diffusion.state_dict(),
    #                 'opt': opt.state_dict(),
    #                 'loss': train_loss_list
    #             }, save_path)
    #             print('model saved!')

    # # get sample sequence from final model
    # sample = diffusion.sample_diffusion_sequence(1, device=device)
    
    # plt.figure(figsize=(20,8))
    # for i in range(1,11):
    #     plt.subplot(1,10,i)
    #     plt.imshow(sample[(i-1)*(time_step)//10][0,0,:,:,image_size//2], cmap='gray')
    # plt.savefig(os.path.join(result_dir, "sample_diffusion_sequence"))
    
    # save loss curve
    plt.figure(figsize=(9,9))
    log_x = np.arange(1, num_iteration+1)
    log_y = train_loss_list
    plt.plot(log_x, log_y)
    plt.savefig(os.path.join(log_dir, 'log'))
    
    
# %%
