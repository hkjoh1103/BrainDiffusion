# %%
# Library
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# get beta scheduler
def get_schedule(args, s=0.008):
    T = args.time_step
    
    if args.schedule == 'cosine':
        def f(t, T):
            return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
        alphas = []
        f0 = f(0, T)
        
        for t in range(T + 1):
            alphas.append(f(t, T) / f0)
            
        betas = []
        
        for t in range(1, T + 1):
            betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
            
        return np.array(betas)
    
    else:
        low = args.schedule_low * 1000 / T
        high = args.schedule_high * 1000 / T
        
        return np.linspace(low, high, T)

# get normalization layer as norm parameter
def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm3d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm3d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

# exponential moving average class
class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)
            
# extract
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion3D():
    def __init__(self, model, args):
        self.model = model
        self.use_ema = args.use_ema
        if self.use_ema:
            self.ema_model = deepcopy(self.model)
            self.ema = EMA(args.ema_decay)
            self.ema_decay = args.ema_decay
            self.ema_update_rate = args.ema_update_rate
            self.ema_start = args.num_iteration // 2
        
        self.image_size = args.image_size
        self.time_step = args.time_step
        
        self.betas = torch.tensor(get_schedule(args, s=0.008)).type(torch.FloatTensor)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)
        self.reciprocal_sqrt_alphas = np.sqrt(1 / self.alphas)
        self.remove_noise_coeff = self.betas / self.sqrt_one_minus_alphas_cumprod
        self.sigma = np.sqrt(self.betas)
        
    def update_ema(self, iteration):
        if not self.use_ema:
            raise ValueError("EMA model is requested to update, but use_ema is False")
        
        if iteration % self.ema_update_rate == 0:
            if iteration < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)
        
    @torch.no_grad()
    def remove_noise(self, x, t, y=None):
        device = x.device
        use_ema = self.use_ema
        
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff.to(device), t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas.to(device), t, x.shape)
            )
        
        else:
            return (
                (x - extract(self.remove_noise_coeff.to(device), t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas.to(device), t, x.shape)
            )

    @torch.no_grad()
    def sample(self, batch_size, device, y=None):
        use_ema = self.use_ema
        x = torch.randn(batch_size, 1, *self.image_size, device=device)
        
        for t in range(self.time_step - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            y_batch = torch.tensor([y], device=device).repeat(batch_size) if y is not None else None
            x = self.remove_noise(x, t_batch, y_batch, use_ema=use_ema)

            if t > 0:
                x += extract(self.sigma.to(device), t_batch, x.shape) * torch.randn_like(x)
        
        del t_batch, y_batch
        
        return x.detach().cpu()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None):
        use_ema = self.use_ema
        x = torch.randn(batch_size, 1, *self.image_size, device=device)
        diffusion_sequence = [x.cpu().detach()]
        
        for t in range(self.time_step - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            y_batch = torch.tensor([y], device=device).repeat(batch_size) if y is not None else None
            x = self.remove_noise(x, t_batch, y_batch, use_ema=use_ema)

            if t > 0:
                x += extract(self.sigma.to(device), t_batch, x.shape) * torch.randn_like(x)
            
            diffusion_sequence.append(x.cpu().detach())
        
        del x, t_batch, y_batch
        
        return diffusion_sequence
    