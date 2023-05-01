# %%
# Library
import numpy as np
import math
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.normalization import GroupNorm

from diffusion_UKB.function import *
from diffusion_UKB.layer import *

# %%
class Unet3D(nn.Module):
    '''
    input : 
        x: tensor (N, in_channels, H, W, D)
        y: None
    Output :
        tensor (N, out_channels, H, W, D)
    '''
    
    def __init__(
        self,
        in_channels,
        base_channels,
        channel_mults,
        num_res_blocks,
        time_emb_dim,
        time_emb_scale=1.0,
        num_classes=None,
        age_positional=False,
        activation=F.silu,
        dropout=0.1,
        attention_resolutions=(),
        norm='gn',
        num_groups=32,
    ):
        super().__init__()
        
        # define self parameters
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.activation = activation
        
        # define time embeding layer
        self.time_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, time_emb_scale),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if time_emb_dim is not None else None
        
        self.age_mlp = nn.Sequential(
            PositionalEmbedding(base_channels, scale=1.0),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        ) if age_positional else None
        
        # define initnial 3D convolutional layer
        self.init_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # initialize down & up phase module list
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # initialize channel lists
        channels = [base_channels]
        now_channels = base_channels
        
        # build downs and channels as channel_mults
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            
            # add ResidualBlock for each level
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    age_positional=age_positional,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                channels.append(now_channels)
            
            # add Downsample for each level except final level
            if i != len(channel_mults)-1:
                self.downs.append(Downsample(now_channels))
                channels.append(now_channels)
            
        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                age_positional=age_positional,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=True,
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                time_emb_dim=time_emb_dim,
                num_classes=num_classes,
                age_positional=age_positional,
                activation=activation,
                norm=norm,
                num_groups=num_groups,
                use_attention=False,
            ),
        ])
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            
            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    time_emb_dim=time_emb_dim,
                    num_classes=num_classes,
                    age_positional=age_positional,
                    activation=activation,
                    norm=norm,
                    num_groups=num_groups,
                    use_attention=i in attention_resolutions,
                ))
                now_channels = out_channels
                
            if i != 0:
                self.ups.append(Upsample(now_channels))
                
        assert len(channels) == 0
        
        self.out_norm = get_norm(norm, base_channels, num_groups)
        self.out_conv = nn.Conv3d(base_channels, in_channels, 3, padding=1)
        
    def forward(self, x, time=None, y=None):
        # acquire time embedding
        if self.time_mlp is not None:
            if time is None:
                # raise ValueError("time conditioning was specified but tim is not passed")
                time = torch.randint(0, 300, (1,), device=x.device)
            
            time_emb = self.time_mlp(time)
            
        else:
            time_emb = None
                    
        # check conditioning parameter is available
        if y is not None:
            if self.num_classes is None:
                raise ValueError("class conditioning is not specified but y is passed")
            if self.age_mlp is not None:
                y = self.age_mlp(y)
        # if self.num_classes is not None and y is None:
        #     raise ValueError("class conditioning was specified but y is not passed")

        # x = (N, 1, H, W, D)
        x = self.init_conv(x)
        
        skips = [x]
        
        for layer in self.downs:
            x = layer(x, time_emb, y)
            skips.append(x)
 
        for layer in self.mid:
            x = layer(x, time_emb, y)
   
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb, y)

        x = self.activation(self.out_norm(x))
        x = self.out_conv(x)
        
        return x


# class GaussianDiffusion3D(nn.Module):
#     __doc__ = r"""Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.

#     Input:
#         x: tensor of shape (N, img_channels, H, W, D)
#         y: tensor of shape (N)
#     Output:
#         scalar loss tensor
#     Args:
#         model (nn.Module): model which estimates diffusion noise
#         img_size (tuple): image size tuple (H, W)
#         img_channels (int): number of image channels
#         betas (np.ndarray): numpy array of diffusion betas
#         loss_type (string): loss type, "l1" or "l2"
#         ema_decay (float): model weights exponential moving average decay
#         ema_start (int): number of steps before EMA
#         ema_update_rate (int): number of steps before each EMA update
#     """
#     def __init__(
#         self,
#         model,
#         img_size,
#         img_channels,
#         num_classes,
#         betas,
#         loss_type="l2",
#         ema_decay=0.9999,
#         ema_start=5000,
#         ema_update_rate=1,
#     ):
#         super().__init__()

#         self.model = model
#         '''
#         # variables for EMA model
#         self.ema_model = deepcopy(model)
#         self.ema = EMA(ema_decay)
#         self.ema_decay = ema_decay
#         self.ema_start = ema_start
#         self.ema_update_rate = ema_update_rate
#         self.step = 0
#         '''

#         self.img_size = img_size
#         self.img_channels = img_channels
#         self.num_classes = num_classes

#         if loss_type not in ["l1", "l2"]:
#             raise ValueError("__init__() got unknown loss type")

#         self.loss_type = loss_type
#         self.num_timesteps = len(betas)

#         alphas = 1.0 - betas
#         alphas_cumprod = np.cumprod(alphas)

#         to_torch = partial(torch.tensor, dtype=torch.float32)

#         self.register_buffer("betas", to_torch(betas))
#         self.register_buffer("alphas", to_torch(alphas))
#         self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

#         self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
#         self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
#         self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

#         self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
#         self.register_buffer("sigma", to_torch(np.sqrt(betas)))

#     '''
#     def update_ema(self):
#         self.step += 1
#         if self.step % self.ema_update_rate == 0:
#             if self.step < self.ema_start:
#                 self.ema_model.load_state_dict(self.model.state_dict())
#             else:
#                 self.ema.update_model_average(self.ema_model, self.model)
#     '''


#     @torch.no_grad()
#     def remove_noise(self, x, t, y):
#         return (
#             (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
#             extract(self.reciprocal_sqrt_alphas, t, x.shape)
#         )

#     @torch.no_grad()
#     def sample(self, batch_size, device, y=None):
#         if y is not None and batch_size != len(y):
#             raise ValueError("sample batch size different from length of given y")

#         x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
#         for t in range(self.num_timesteps - 1, -1, -1):
#             t_batch = torch.tensor([t], device=device).repeat(batch_size)
#             x = self.remove_noise(x, t_batch, y)

#             if t > 0:
#                 x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        
#         return x.cpu().detach()

#     @torch.no_grad()
#     def sample_diffusion_sequence(self, batch_size, device, y=None):
#         if y is not None and batch_size != len(y):
#             raise ValueError("sample batch size different from length of given y")

#         x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
#         diffusion_sequence = [x.cpu().detach()]
        
#         for t in range(self.num_timesteps - 1, -1, -1):
#             t_batch = torch.tensor([t], device=device).repeat(batch_size)
#             x = self.remove_noise(x, t_batch, y)

#             if t > 0:
#                 x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            
#             diffusion_sequence.append(x.cpu().detach())
        
#         return diffusion_sequence

#     def forward(self, x, y=None):
#         b, c, h, w, d = x.shape
#         device = x.device

#         if h != self.img_size[0]:
#             raise ValueError("image height does not match diffusion parameters")
#         if w != self.img_size[0]:
#             raise ValueError("image width does not match diffusion parameters")
#         if d != self.img_size[0]:
#             raise ValueError("image depth does not match diffusion parameters")
        
#         # random creation of time step
#         t = torch.randint(0, self.num_timesteps, (b,), device=device)
        
#         # gaussian noise generation
#         noise = torch.randn_like(x)
        
#         # x_t = noised x at (t+1) time step
#         x_t = extract(self.sqrt_alphas_cumprod, t, x.shape) * x + \
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        
#         # noise estimation
#         estimated_noise = self.model(x_t, t, y)
        
#         if self.loss_type == "l1":
#             loss = F.l1_loss(estimated_noise, noise)
#         elif self.loss_type == "l2":
#             loss = F.mse_loss(estimated_noise, noise)
            
#         return loss