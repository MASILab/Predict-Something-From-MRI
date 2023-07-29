# Evaluate the models on the validation set (of the fold), 
# testing set (healthy), and testing set (impaired).
# 
# Author: Chenyu Gao
# Date: Jul 29, 2023

import torch
import pdb
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from model import MLP
from dataset import ROIBasedAgePredictionDataset_testing
from functions import make_prediction
from torch.utils.data import DataLoader

save_evaluation_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation')
path_train_all_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_baseline_ROI-based-measure/data/ROIbased_measure_train.csv'
path_test_healthy_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_baseline_ROI-based-measure/data/ROIbased_measure_test_healthy.csv'
path_test_impaired_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_baseline_ROI-based-measure/data/ROIbased_measure_test_impaired.csv'

# Trained models that are available for testing
path_weights_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/weights')
list_model_name = [fn.name for fn in path_weights_root.iterdir() if fn.name.startswith('MLP-')]
list_hidden_layer_sizes = []
for name in list_model_name:
    hidden_layer_sizes = [int(x) for x in name.replace('MLP-','').split('-')]
    list_hidden_layer_sizes.append(hidden_layer_sizes)

for i, model_name in enumerate(list_model_name):
    hidden_layer_sizes = list_hidden_layer_sizes[i]
    for fold_id in [1,2,3,4,5]:
        
        # Load selected model and weights (lowest validation loss)
        model = MLP(hidden_layer_sizes=hidden_layer_sizes)
        path_weights_model = path_weights_root / model_name / f"fold-{fold_id}"
        list_pth_file = [fn for fn in path_weights_model.iterdir() if fn.name.endswith('.pth')]
        list_valloss = [float(fn.name.split('_valloss-')[1].replace('.pth','')) for fn in path_weights_model.iterdir() if fn.name.endswith('.pth')]
        path_pth = list_pth_file[list_valloss.index(min(list_valloss))]
        
        checkpoint = torch.load(path_pth)
        model.load_state_dict(checkpoint)

        #### Report 1: Confirm the performance on the ENTIRE validation set (instead of random sampling for each subject during training)
        df_train_all = pd.read_csv(path_train_all_csv)
        df_train_all = df_train_all.fillna(df_train_all.median(numeric_only=True))
        df_train_all.reset_index(drop=True, inplace=True)
        df_train_all_encoded = pd.get_dummies(df_train_all, columns=['Sex'], drop_first=True)

        subjects_validation = np.load(f"/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train_subjects_v_fold_{fold_id}.npy",
                                      allow_pickle=True)  # subject-level splitting of train/val
        df_validation = df_train_all_encoded[df_train_all_encoded['Subject'].isin(subjects_validation)]

        df_validation_with_prediction = make_prediction(model, df_validation)
        
        save_to = save_evaluation_root / model_name / f'fold-{fold_id}'
        subprocess.run(['mkdir', '-p', save_to])
        df_validation_with_prediction = df_validation_with_prediction[['Dataset','Subject','Session','Sample','Age','Diagnosis','Sex_male','Age_gt','Age_predicted']]
        df_validation_with_prediction.to_csv((save_to / f'{model_name}_fold-{fold_id}_prediction_validation.csv'), index=False)

        #### Report 2: Test- healthy subjects
        df_test_healthy = pd.read_csv(path_test_healthy_csv)
        df_test_healthy = df_test_healthy.fillna(df_test_healthy.median(numeric_only=True))
        df_test_healthy.reset_index(drop=True, inplace=True)
        df_test_healthy_encoded = pd.get_dummies(df_test_healthy, columns=['Sex'], drop_first=True)

        df_test_healthy_with_prediction = make_prediction(model, df_test_healthy_encoded)
        df_test_healthy_with_prediction = df_test_healthy_with_prediction[['Dataset','Subject','Session','Sample','Age','Diagnosis','Sex_male','Age_gt','Age_predicted']]
        df_test_healthy_with_prediction.to_csv((save_to / f'{model_name}_fold-{fold_id}_prediction_test_healthy.csv'), index=False)
        
        #### Report 2: Test- impaired subjects
        df_test_impaired = pd.read_csv(path_test_impaired_csv)
        df_test_impaired = df_test_impaired.fillna(df_test_impaired.median(numeric_only=True))
        df_test_impaired.reset_index(drop=True, inplace=True)
        df_test_impaired_encoded = pd.get_dummies(df_test_impaired, columns=['Sex'], drop_first=True)
        
        df_test_impaired_with_prediction = make_prediction(model, df_test_impaired_encoded)
        df_test_impaired_with_prediction = df_test_impaired_with_prediction[['Dataset','Subject','Session','Sample','Age','Diagnosis','Sex_male','Age_gt','Age_predicted']]
        df_test_impaired_with_prediction.to_csv((save_to/f'{model_name}_fold-{fold_id}_prediction_test_impaired.csv'), index=False)