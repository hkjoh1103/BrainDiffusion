# %%
# Library
import torch

import os

import matplotlib.pyplot as plt

from diffusion_SNU.function import GaussianDiffusion3D, image_print, image_save
from diffusion_SNU.model import Unet3D

# %%
def sample(args):
    mode = args.mode
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
        
    image_size = args.image_size
    base_channels = args.base_channels
    channel_mults = args.channel_mults
    num_res_blocks = args.num_res_blocks
    time_emb_dim = args.time_emb_dim
    
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    print('mode : ', mode)
    print('experiment no. : ', name)
    
    base_dir = f'./experiment/{name}'
    ckpt_dir = os.path.join(base_dir, 'checkpoint')
    sample_dir = os.path.join(base_dir, 'sample')
    ckpt_list = os.listdir(ckpt_dir)
    
    if not os.path.exists(ckpt_dir):
        raise KeyError(f'{name} or {name}/checkpoint do not exist')
    if not ckpt_list:
        raise KeyError(f'{name}/checkpoint does not have any model file')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    model = Unet3D(
        1,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
        num_classes=age_classes,
        age_positional=age_positional,
    )
    
    load = torch.load(os.path.join(ckpt_dir, ckpt_list[-1]), map_location='cpu')
    model.load_state_dict(load['net'])
    del load
    
    model.to(device)
    diffusion = GaussianDiffusion3D(model, args)
    
    with torch.no_grad():
        sample1 = diffusion.sample(batch_size=4, device=device, y=40)
        image_print(sample1)
        image_save(sample1, sample_dir, '_age40')
        
        sample2 =diffusion.sample(batch_size=4, device=device, y=60)
        image_print(sample2)
        image_save(sample2, sample_dir, '_age60')
        
    print('sampling finished!!')
# %%
