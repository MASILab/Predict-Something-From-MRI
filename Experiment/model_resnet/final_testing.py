# Select model for final testing.
# See Figure/fig_complexity_performance for why I choose this model.
# 
# Author: Chenyu Gao
# Date: Jul 25, 2023

import torch
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import *
from train import select_model
from dataset import AgePredictionDataset, AgePredictionDataset_testing
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load selected model and weights
model_name = 'resnet18_MLP_64'
path_weights = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/weights/resnet18_MLP_64/fold-5/model_fold-5_epoch-38_valloss-31.9250.pth'
model = select_model(model_name=model_name)
checkpoint = torch.load(path_weights)
model.load_state_dict(checkpoint)
model.to(device)

# Validation and Testing Sets
path_subjects_validation = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train_subjects_v_fold_5.npy'
path_train_all_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train.csv'
path_test_healthy_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/test_healthy.csv'
path_test_impaired_csv = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/test_impaired.csv'

# Confirm the performance on the ENTIRE validation set (instead of random sample for each subject)
df_train_all = pd.read_csv(path_train_all_csv)
subjects_validation = np.load(path_subjects_validation, allow_pickle=True)  # subject-level splitting of train/val
df_validation = df_train_all[df_train_all['Subject'].isin(subjects_validation)]

dataset_validation = AgePredictionDataset_testing(df_validation)
dl_val = DataLoader(dataset_validation, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

model.eval()
gt = torch.tensor([]).to(device)
prediction = torch.tensor([]).to(device)

with torch.no_grad():
    for _, fa_val, md_val, sex_val, age_val in tqdm(dl_val):
        fa_val, md_val, sex_val, age_val = fa_val.to(device, non_blocking=True), md_val.to(device, non_blocking=True), sex_val.to(device, non_blocking=True), age_val.to(device, non_blocking=True)

        input_img_val = torch.cat((fa_val,md_val), dim=1)
        prediction = torch.cat((prediction, model(input_img_val, sex_val)))
        gt = torch.cat((gt, age_val.view(-1, 1)))

loss_mse = torch.nn.MSELoss()
loss_l1 = torch.nn.L1Loss()
print(f"Confirmation: Validation\tLoss_MSE: {loss_mse(prediction, gt).item()}\tLoss_L1: {loss_l1(prediction, gt).item()}")
print("If the MSE is around (or below) 33, it's good.")

# Test: healthy subjects
df_test_healthy = pd.read_csv(path_test_healthy_csv)
dataset_test_healthy = AgePredictionDataset_testing(df_test_healthy)
dl_test_healthy = DataLoader(dataset_test_healthy, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

model.eval()
# record performance
list_df_session_id = []
list_df_sample_id = []
gt = torch.tensor([]).to(device)
prediction = torch.tensor([]).to(device)

with torch.no_grad():
    for sample_info, fa, md, sex, age in tqdm(dl_test_healthy):
        
        fa, md, sex, age = fa.to(device, non_blocking=True), md.to(device, non_blocking=True), sex.to(device, non_blocking=True), age.to(device, non_blocking=True)
        
        input_img = torch.cat((fa,md), dim=1)
        output = model(input_img, sex)

        prediction = torch.cat((prediction, output))
        gt = torch.cat((gt, age.view(-1, 1)))
        list_df_session_id += sample_info[0]
        list_df_sample_id += sample_info[1]

loss_mse = torch.nn.MSELoss()
loss_l1 = torch.nn.L1Loss()
print(f"Testing (healthy)\tLoss_MSE: {loss_mse(prediction, gt).item()}\tLoss_L1: {loss_l1(prediction, gt).item()}")

list_df_age_gt = torch.squeeze(gt).tolist()
list_df_age_predict = torch.squeeze(prediction).tolist()

# write into spreadsheet
df_test_healthy['Age_gt'] = None
df_test_healthy['Age_predicted'] = None

for i in range(len(list_df_session_id)):
    df_test_healthy.loc[(df_test_healthy['Session']==list_df_session_id[i])&(df_test_healthy['Sample']==list_df_sample_id[i]), 
                        'Age_gt'] = list_df_age_gt[i]
    df_test_healthy.loc[(df_test_healthy['Session']==list_df_session_id[i])&(df_test_healthy['Sample']==list_df_sample_id[i]), 
                        'Age_predicted'] = list_df_age_predict[i]
    
df_test_healthy.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/prediction_resnet18_MLP_64/prediction_test_healthy.csv', index=False)


# Test: impaired subjects
df_test_impaired = pd.read_csv(path_test_impaired_csv)
dataset_test_impaired = AgePredictionDataset_testing(df_test_impaired)
dl_test_impaired = DataLoader(dataset_test_impaired, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

model.eval()
# record performance
list_df_session_id = []
list_df_sample_id = []
gt = torch.tensor([]).to(device)
prediction = torch.tensor([]).to(device)

with torch.no_grad():
    for sample_info, fa, md, sex, age in tqdm(dl_test_impaired):
        
        fa, md, sex, age = fa.to(device, non_blocking=True), md.to(device, non_blocking=True), sex.to(device, non_blocking=True), age.to(device, non_blocking=True)
        
        input_img = torch.cat((fa,md), dim=1)
        output = model(input_img, sex)

        prediction = torch.cat((prediction, output))
        gt = torch.cat((gt, age.view(-1, 1)))
        list_df_session_id += sample_info[0]
        list_df_sample_id += sample_info[1]

loss_mse = torch.nn.MSELoss()
loss_l1 = torch.nn.L1Loss()
print(f"Testing (impaired)\tLoss_MSE: {loss_mse(prediction, gt).item()}\tLoss_L1: {loss_l1(prediction, gt).item()}")

list_df_age_gt = torch.squeeze(gt).tolist()
list_df_age_predict = torch.squeeze(prediction).tolist()

# write into spreadsheet
df_test_impaired['Age_gt'] = None
df_test_impaired['Age_predicted'] = None

for i in range(len(list_df_session_id)):
    df_test_impaired.loc[(df_test_impaired['Session']==list_df_session_id[i])&(df_test_impaired['Sample']==list_df_sample_id[i]), 
                         'Age_gt'] = list_df_age_gt[i]
    df_test_impaired.loc[(df_test_impaired['Session']==list_df_session_id[i])&(df_test_impaired['Sample']==list_df_sample_id[i]),
                         'Age_predicted'] = list_df_age_predict[i]
    
df_test_impaired.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/prediction_resnet18_MLP_64/prediction_test_impaired.csv', index=False)
