# %%
# Library
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

import torchio as tio
import monai

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchsummary import summary

from diffusion.data import DataPreprocessing
from diffusion.model import Unet3D, GaussianDiffusion3D
from diffusion.function import *
from diffusion.util import *

# %%
def train(gpu_num, args):
    # define parameters from arguments
    name = args.name
    
    age_classes = None
    age_positional = False
    
    if args.age_type == 'int':
        age_classes = 100
    elif args.age_type == 'cat':
        age_classes = 5
    elif args.age_type == 'pos':
        age_classes = 100
        age_positional = True
    
    use_multiGPU = args.use_multiGPU
    rank = args.machine_id * args.num_gpu_processes + gpu_num
    world_size = args.num_gpu_processes * args.num_machines

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
    base_dir = f'./experiment/{name}'
    ckpt_dir = os.path.join(base_dir, 'checkpoint')
    log_dir = os.path.join(base_dir, 'log')
    result_dir = os.path.join(base_dir, 'result')
    
    log_rate = args.log_rate
    save_rate = args.save_rate
    
    if use_multiGPU:
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    
    print('device count: %d' %(torch.cuda.device_count()))
    
    torch.cuda.set_device(gpu_num)
    device = torch.device(gpu_num) if torch.cuda.is_available() else torch.device("cpu")
    
    print('device : %s' %device)
    print('='*30)
    
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
    time_before = time.time()
    data = DataPreprocessing(args, rank=rank, world_size=world_size)
    
    dataloader = data.get_dataloader()
    print('size of dataloader : %d' %(len(dataloader)))
    dataloader = cycle(dataloader)
 
    # define model & optimizer
    model = Unet3D(
        1,
        base_channels,
        channel_mults,
        num_res_blocks,
        time_emb_dim,
        num_classes=age_classes,
        age_positional=age_positional,
    )
    
    # print(summary(model, input_size=((1, *image_size),1,1), device='cpu'))
    
    betas = get_schedule(args, s=0.008)
    
    # check sample image and noise sequence
    x = next(dataloader)
    sample_data = x['image']['data'][0, 0, :, :, :]
    
    plt.figure(figsize=(15,6))
    sample_subject = tio.Subject(
        id=x['id'][0],
        image=tio.ScalarImage(tensor=x['image']['data'][0])
    )
    sample_subject.plot(show=False, output_path=os.path.join(result_dir, "sample_subject.png"))

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
        plt.imshow(sample_sequence[(i-1)*(time_step)//10][:, :, image_size[2]//2], cmap='gray')
    plt.savefig(os.path.join(result_dir, "sample_noise_sequence.png"))
    
    print('='*30)
    print('initial sampling time : %.1f' %(time.time() - time_before))
    
    # define variables related to diffusion process
    to_torch = partial(torch.tensor, dtype=torch.float32)
    
    betas = to_torch(betas)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)
    reciprocal_sqrt_alphas = np.sqrt(1 / alphas)
    remove_noise_coeff = betas / np.sqrt(1 - alphas_cumprod)
    sigma = np.sqrt(betas)

    def get_loss(model, x, y=None):
        b, c, h, w, d = x.shape
        device = x.device
        
        if h != image_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != image_size[1]:
            raise ValueError("image width does not match diffusion parameters")
        if d != image_size[2]:
            raise ValueError("image depth does not match diffusion parameters")
        
        # random creation of time step
        t = torch.randint(0, time_step, (b,), device=device)
        
        # gaussian noise generation
        noise = torch.randn_like(x)
        
        # x_t = noised x at (t+1) time step
        x_t = extract(sqrt_alphas_cumprod.to(device), t, x.shape) * x + \
            extract(sqrt_one_minus_alphas_cumprod.to(device), t, x.shape) * noise
        
        # noise estimation
        estimated_noise = model(x_t, t, y)

        loss = F.mse_loss(estimated_noise, noise)
            
        return loss
        
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
    model.to(device)   
    if use_multiGPU:
        model = DDP(model, device_ids=[gpu_num], find_unused_parameters=True)
        
    diffusion = GaussianDiffusion3D(model, args)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    train_loss_list = []
    iteration = 1
    time_start = time.time()
    
    while iteration <= num_iteration:
        dataset = next(dataloader)
        data = dataset['image']['data'].to(device, non_blocking=True)
        if args.age_type in ['int', 'pos']:
            age = dataset['age'].to(device, non_blocking=True)
        elif args.age_type == 'cat':
            age = dataset['age_cat'].to(device, non_blocking=True)
        else:
            age = None
        
        loss = get_loss(model=model, x=data, y=age)

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if args.use_ema:
            diffusion.update_ema(iteration=iteration)

        train_loss_list.append(loss.item())

        
        if (iteration) % log_rate == 0 and rank == 0:
            # calculate running loss
            
            print("Iteration %d / %d | loss %.4f | time elapsed %.1f sec"
                  %(iteration,
                    num_iteration,
                    np.mean(train_loss_list[-log_rate:]),
                    time.time() - time_start)
            )
            
        
        if (iteration) % save_rate == 0 and rank == 0:
            with torch.no_grad():
                # save nii file
                model.eval()
                generated_data = diffusion.sample(4, device=device, y=40)
                
                b,*_ = generated_data.shape
                plt.figure(figsize=(9, 3*b))
                for i in range(b):
                    plt.subplot(b, 3, 3*i + 1)
                    plt.imshow(generated_data[i,0,image_size[0]//2,:,:], cmap='gray')
                    plt.title('sagital')
                    
                    plt.subplot(b, 3, 3*i + 2)
                    plt.imshow(generated_data[i,0,:,image_size[1]//2,:], cmap='gray')
                    plt.title('coronal')
                    
                    plt.subplot(b, 3, 3*i + 3)
                    plt.imshow(generated_data[i,0,:,:,image_size[2]//2], cmap='gray')
                    plt.title('axial')
                    
                plt.savefig(os.path.join(result_dir, f"Iteration{iteration}_sample_age40.png"))
                
                generated_data2 = diffusion.sample(4, device=device, y=60)
                
                b,*_ = generated_data2.shape
                plt.figure(figsize=(9, 3*b))
                for i in range(b):
                    plt.subplot(b, 3, 3*i + 1)
                    plt.imshow(generated_data2[i,0,image_size[0]//2,:,:], cmap='gray')
                    plt.title('sagital')
                    
                    plt.subplot(b, 3, 3*i + 2)
                    plt.imshow(generated_data2[i,0,:,image_size[1]//2,:], cmap='gray')
                    plt.title('coronal')
                    
                    plt.subplot(b, 3, 3*i + 3)
                    plt.imshow(generated_data2[i,0,:,:,image_size[2]//2], cmap='gray')
                    plt.title('axial')
                    
                plt.savefig(os.path.join(result_dir, f"Iteration{iteration}_sample_age60.png"))
                
                #save one pth file
                save_path = os.path.join(ckpt_dir, 'model_test1.pth')
    
                #save pth file as iteration
                #save_path = os.path.join(ckpt_dir, 'model_iteration%d.pth' %i)
                torch.save({
                    'net': model.state_dict(),
                    'opt': opt.state_dict(),
                    'loss': train_loss_list
                }, save_path)
                print('model saved!')
                model.train()
        
        del data, loss
        torch.cuda.empty_cache()
        iteration += 1
    
    with torch.no_grad():
        # get sample sequence from final model
        model.eval()
        generated_sequence = diffusion.sample_diffusion_sequence(4, device=device)
        b,*_ = generated_sequence[0].shape
        
        plt.figure(figsize=(30,3*b))
        for i in range(b):
            for j in range(1,11):
                plt.subplot(b,11,11*i + j)
                plt.imshow(generated_sequence[(j-1)*(time_step)//10][i,0,:,:,image_size[2]//2], cmap='gray')
            plt.subplot(b,11,11*i + 11)
            plt.imshow(generated_sequence[-1][i,0,:,:,image_size[2]//2], cmap='gray')
        
        if rank == 0:
            plt.savefig(os.path.join(result_dir, "sample_diffusion_sequence"))
        
        # save loss curve
        plt.figure(figsize=(9,9))
        log_x = np.arange(0, num_iteration, 10)
        log_y = [np.mean(train_loss_list[i:i+10]) for i in log_x]
        plt.plot(log_x, log_y)
        if rank == 0:
            plt.savefig(os.path.join(log_dir, 'log'))
    
        
# %%
