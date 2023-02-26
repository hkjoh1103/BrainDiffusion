# %%
# Library
import os
from glob import glob
import torchio as tio
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, config, rank=1, world_size=1):
        # self.data_fn = config.data_fn
        self.data_dir = config.data_dir
        self.data_type = self.data_dir.split('/')[-1]
        self.patient_dir = config.patient_dir
        
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        
        if config.mri_type == 'adc':
            self.crop_size = (104,104,72)
        elif config.mri_type == 'flair':
            self.crop_size = (180,240,200)
        elif config.mri_type == 't1':
            self.crop_size = (180,240,200)
        
        self.use_multiGPU = config.use_multiGPU
        self.rank=rank
        self.world_size=world_size
        
    def get_list(self):
        if self.data_type == 'flair':
            file_name = '*_[2-3].nii*'
        elif self.data_type == 't1':
            file_name = '*_[2-3].nii*'
        elif self.data_type == 'dti_md':
            file_name = '*.nii*'

        fn_list = glob(os.path.join(self.data_dir, file_name))
        fn_list = sorted(fn_list)
        
        data_pt = pd.read_csv(self.patient_dir, encoding='utf-8', header=0, index_col=0, usecols=['eid', '21022-0.0']).dropna(axis=0)
        
        subject_list = []
        for subject in fn_list:
            patient_id = subject.split('/')[-1].split('_')[0]
            patient_age = data_pt.loc[int(patient_id)]['21022-0.0'].astype(int)
            tio_subject = tio.Subject(
                id=patient_id,
                image=tio.ScalarImage(subject),
                age=patient_age,
                age_cat=(patient_age//10 - 3),
            )
            subject_list.append(tio_subject)
        
        print('torchio subject list created')
        
        return subject_list
    
    def get_dataset(self):
        fn_list = self.get_list()

        transform = tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(self.crop_size, padding_mode=0),
            tio.Resize(self.image_size),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
            tio.RandomFlip(axes='lr',flip_probability=0.2),
            tio.RandomAffine(translation=(5,5,5), default_pad_value='otsu'),
            # tio.RandomElasticDeformation(num_control_points=6, max_displacement=4)
        ])
        
        dataset = tio.SubjectsDataset(fn_list, transform=transform)
        print('transform completed')
        
        if self.use_multiGPU:
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank
            )
            return dataset, sampler

        else:
            return dataset
    
    def get_dataloader(self):
        if self.use_multiGPU:
            dataset, sampler = self.get_dataset()
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=sampler
            )
            
        else:
            dataset = self.get_dataset()
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2
            )
        print('dataloader completed')
        
        return dataloader
