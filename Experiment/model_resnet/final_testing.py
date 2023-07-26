# Evaluate the models on the validation set (of the fold), 
# testing set (healthy), and testing set (impaired).
# 
# Author: Chenyu Gao
# Date: Jul 26, 2023

import torch
import pdb
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from models import *
from train import select_model
from dataset import AgePredictionDataset_testing
from utls.functions import make_prediction
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_evaluation_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation')
path_train_all_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train.csv'
path_test_healthy_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/test_healthy.csv'
path_test_impaired_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/test_impaired.csv'

# Trained models that are available for testing
path_weights_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/weights')
list_model_name = [fn.name for fn in path_weights_root.iterdir() if fn.name.startswith('resnet')]

for model_name in list_model_name:
    for fold_id in [1,2,3,4,5]:
        
        # Load selected model and weights
        model = select_model(model_name=model_name)
        path_weights_model = path_weights_root / model_name / f"fold-{fold_id}"
        list_pth_file = [fn for fn in path_weights_model.iterdir() if fn.name.endswith('.pth')]
        path_pth = list_pth_file[0]
        
        checkpoint = torch.load(path_pth)
        model.load_state_dict(checkpoint)
        model.to(device)

        #### Report 1: Confirm the performance on the ENTIRE validation set (instead of random sampling for each subject during training)
        df_train_all = pd.read_csv(path_train_all_csv)
        subjects_validation = np.load(f"/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train_subjects_v_fold_{fold_id}.npy",
                                      allow_pickle=True)  # subject-level splitting of train/val
        df_validation = df_train_all[df_train_all['Subject'].isin(subjects_validation)]

        df_validation_with_prediction = make_prediction(model=model, df=df_validation, device=device, batch_size=4)
        
        save_to = save_evaluation_root / model_name / f'fold-{fold_id}'
        subprocess.run(['mkdir', '-p', save_to])
        df_validation_with_prediction.to_csv((save_to / f'{model_name}_fold-{fold_id}_prediction_validation.csv'), index=False)

        #### Report 2: Test- healthy subjects
        df_test_healthy = pd.read_csv(path_test_healthy_csv)
        
        df_test_healthy_with_prediction = make_prediction(model=model, df=df_test_healthy, device=device, batch_size=4)
        df_test_healthy_with_prediction.to_csv((save_to / f'{model_name}_fold-{fold_id}_prediction_test_healthy.csv'), index=False)
        
        #### Report 2: Test- impaired subjects
        df_test_impaired = pd.read_csv(path_test_impaired_csv)
        
        df_test_impaired_with_prediction = make_prediction(model=model, df=df_test_impaired, device=device, batch_size=4)
        df_test_impaired_with_prediction.to_csv((save_to/f'{model_name}_fold-{fold_id}_prediction_test_impaired.csv'), index=False)