import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class ROIBasedAgePredictionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.subjects = df['Subject'].unique()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # select a random sample from this subject
        subject = self.subjects[idx]
        df_subject = self.df[self.df['Subject'] == subject]
        sample = df_subject.sample(n=1, random_state=np.random.randint(0,1000))
        
        X = sample.drop(['Dataset', 'Subject', 'Session', 'Sample', 'Diagnosis', 'Age'], axis=1)  # case sensitive
        y = sample['Age']
        X = torch.tensor(np.squeeze(X.values), dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        # sanity check 
        if (X.shape != (537,)) or (y.shape != (1,)):
            print("Warning: Check the dataloader. The size doesn't match. X: {}\ty: {}".format(X.shape, y.shape))

        return X, y

class ROIBasedAgePredictionDataset_testing(Dataset):
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

        X = sample.drop(['Dataset', 'Subject', 'Session', 'Sample', 'Diagnosis', 'Age'])
        y = sample['Age']
        X = torch.tensor(np.squeeze(X), dtype=torch.float32)
        y = torch.tensor([y], dtype=torch.float32)
        
        # sanity check 
        if (X.shape != (537,)) or (y.shape != (1,)):
            print("Warning: Check the dataloader. The size doesn't match. X: {}\ty: {}".format(X.shape, y.shape))
            
        return sample_info, X, y
