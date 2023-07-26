import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from models import *
from train import select_model
from dataset import AgePredictionDataset_testing
from torch.utils.data import DataLoader


def make_prediction(model, df, device, batch_size=4):
    
    dataset = AgePredictionDataset_testing(df)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()
    
    # record information
    list_df_session_id = []
    list_df_sample_id = []
    gt = torch.tensor([]).to(device)
    prediction = torch.tensor([]).to(device)

    with torch.no_grad():
        for sample_info, fa, md, sex, age in dl:
            fa, md, sex, age = fa.to(device, non_blocking=True), md.to(device, non_blocking=True), sex.to(device, non_blocking=True), age.to(device, non_blocking=True)

            input_img = torch.cat((fa,md), dim=1)
            output = model(input_img, sex)
            
            # record infomation
            prediction = torch.cat((prediction, output))
            gt = torch.cat((gt, age.view(-1, 1)))
            list_df_session_id += sample_info[0]
            list_df_sample_id += sample_info[1]
    
        loss_mse = torch.nn.MSELoss()
        loss_l1 = torch.nn.L1Loss()
        print(f"Loss_MSE: {loss_mse(prediction, gt).item()}\tLoss_L1: {loss_l1(prediction, gt).item()}")

    # write information into spreadsheet
    list_df_age_gt = torch.squeeze(gt).tolist()
    list_df_age_predict = torch.squeeze(prediction).tolist()
    
    # actually both give warnings
    # df.loc[:, 'Age_gt'] = None
    # df.loc[:, 'Age_predicted'] = None
    df['Age_gt'] = None
    df['Age_predicted'] = None

    for i in range(len(list_df_session_id)):
        df.loc[(df['Session']==list_df_session_id[i])&(df['Sample']==list_df_sample_id[i]),
               'Age_gt'] = list_df_age_gt[i]
        df.loc[(df['Session']==list_df_session_id[i])&(df['Sample']==list_df_sample_id[i]), 
               'Age_predicted'] = list_df_age_predict[i]
    
    return df