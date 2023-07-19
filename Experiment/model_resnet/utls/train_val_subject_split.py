# 5-fold splitting of the subjects in the "training set" (../data/train.csv) 
# into training and validation set.
# For reproducibility, I record the subjects of each split into files.
# 
# Author: Chenyu Gao
# Date: Jul 19, 2023

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

random_seed = 42
train_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train.csv'
df = pd.read_csv(train_csv)

# all subjects
subjects = df['Subject'].unique()
subject_indices = np.arange(len(subjects))

kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

for i, (train_indices, val_indices) in enumerate(kf.split(subject_indices)):
    subjects_train, subjects_val = subjects[train_indices], subjects[val_indices]
    
    np.save(f'/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train_subjects_t_fold_{i+1}.npy', subjects_train)
    np.save(f'/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train_subjects_v_fold_{i+1}.npy', subjects_val)

