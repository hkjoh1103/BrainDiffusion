# %%
# Library
import numpy as np

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