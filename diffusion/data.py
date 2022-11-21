# %%
# Library
import os
from glob import glob

from torch.utils.data import Dataset, DataLoader

import monai
from monai.transforms import (
    Compose, LoadImage, LoadImaged,
    EnsureChannelFirst, ResizeWithPadOrCrop, EnsureType,
    Resize
)

# %%
# define Datasets class
class Datasets(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        return self.df[i] #tensor.size(1, 256, 256, 256)
    
# %%
# define DataPreprocessing class
class DataPreprocessing():
    def __init__(self, config):
        # self.data_fn = config.data_fn
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        
    def get_list(self):
        fn_list = glob(os.path.join(self.data_dir, '*_[2-3].nii*'))
        
        return sorted(fn_list)
    
    def get_dataset(self):
        fn_list = self.get_list()
        
        transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(spatial_size=[64,64,64]),
            EnsureType(data_type='tensor')
        ])
        
        dataset = Datasets(transform(fn_list))
        
        return dataset
    
    def get_dataloader(self):
        dataloader = DataLoader(self.get_dataset(), batch_size=self.batch_size, shuffle=True, num_workers=2)
        
        return dataloader