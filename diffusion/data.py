# %%
# Library
import os
from glob import glob
import torchio as tio

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
    def __init__(self, config):
        # self.data_fn = config.data_fn
        self.data_dir = config.data_dir
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        
    def get_list(self):
        fn_list = glob(os.path.join(self.data_dir, '*_[2-3].nii*'))
        fn_list = sorted(fn_list)
        
        subject_list = []
        for subject in fn_list:
            tio_subject = tio.Subject(
                id=subject.split('/')[-1].split('_')[0],
                flair=tio.ScalarImage(subject),
            )
            subject_list.append(tio_subject)
        
        print('torchio subject list created')
        
        return subject_list
    
    def get_dataset(self):
        fn_list = self.get_list()
        image_size = self.image_size

        transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resize([image_size, image_size, image_size]),
            tio.RescaleIntensity(out_min_max=(-1, 1)),
        ])
        
        dataset = tio.SubjectsDataset(fn_list, transform=transform)
        
        print('transform completed')

        return dataset
    
    def get_dataloader(self):
        dataloader = DataLoader(self.get_dataset(), batch_size=self.batch_size, shuffle=True, num_workers=2)
        print('dataloader completed')
        
        return dataloader
