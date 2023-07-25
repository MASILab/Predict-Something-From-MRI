import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord, EnsureChannelFirstd, ResizeD
import pandas as pd
import numpy as np
import random
from pathlib import Path

class AgePredictionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.subjects = df['Subject'].unique()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        df_subject = self.df[self.df['Subject'] == subject]

        # select a random sample from this subject
        sample = df_subject.sample(n=1, random_state=np.random.randint(0,100))
        
        dir = Path(sample['dir'].values[0])
        img_fa = dir / 'dwmri%fa_brain_MNI152_linear.nii.gz'
        img_md = dir / 'dwmri%md_brain_MNI152_linear.nii.gz'
        dict_sex2numeric = {'female':0, 'male':1}
        sex = torch.tensor(dict_sex2numeric[sample['Sex'].values[0]], dtype=torch.float32)
        age = torch.tensor(sample['Age'].values[0], dtype=torch.float32)

        # Transform
        transform = Compose([
            LoadImaged(keys=['fa', 'md'], image_only=True),
            EnsureChannelFirstd(keys=['fa', 'md']),
            ResizeD(keys=['fa', 'md'], spatial_size=(128, 128, 128), mode='area'),
            ToTensord(keys=['fa', 'md'])
        ])
        data_dict = {'fa': img_fa, 'md': img_md}
        data_dict = transform(data_dict)
        
        return data_dict['fa'], data_dict['md'], sex, age

class AgePredictionDataset_testing(Dataset):
    """Dataloader for model inferencing. 
    The difference is that all samples will be covered. There is no randomness.

    """
    
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        
        sample_info = (sample['Session'], sample['Sample'])
        
        dir = Path(sample['dir'])
        img_fa = dir / 'dwmri%fa_brain_MNI152_linear.nii.gz'
        img_md = dir / 'dwmri%md_brain_MNI152_linear.nii.gz'
        dict_sex2numeric = {'female':0, 'male':1}
        sex = torch.tensor(dict_sex2numeric[sample['Sex']], dtype=torch.float32)
        age = torch.tensor(sample['Age'], dtype=torch.float32)

        # Transform
        transform = Compose([
            LoadImaged(keys=['fa', 'md'], image_only=True),
            EnsureChannelFirstd(keys=['fa', 'md']),
            ResizeD(keys=['fa', 'md'], spatial_size=(128, 128, 128), mode='area'),
            ToTensord(keys=['fa', 'md'])
        ])
        data_dict = {'fa': img_fa, 'md': img_md}
        data_dict = transform(data_dict)
        
        return sample_info, data_dict['fa'], data_dict['md'], sex, age
