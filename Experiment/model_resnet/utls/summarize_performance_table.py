# Generate the model performance table for the SPIE paper.
# Author: Chenyu Gao
# Date: Jul 27, 2023

import pandas as pd

df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/summary_model_performance.csv')
# print(df)
for full_name in df['Full_Name'].unique():
    val_mean = df.loc[df['Full_Name']==full_name, 'Val_MAE'].mean()
    val_std = df.loc[df['Full_Name']==full_name, 'Val_MAE'].std()
    
    test_healthy_mean = df.loc[df['Full_Name']==full_name, 'Test_Healthy_MAE'].mean()
    test_healthy_std = df.loc[df['Full_Name']==full_name, 'Test_Healthy_MAE'].std()
    
    test_impaired_mean = df.loc[df['Full_Name']==full_name, 'Test_Impaired_MAE'].mean()
    test_impaired_std = df.loc[df['Full_Name']==full_name, 'Test_Impaired_MAE'].std()
    
    print(f'Model: {full_name}\tValidation: {val_mean:.2f}±{val_std:.2f}\tTest (Healthy): {test_healthy_mean:.2f}±{test_healthy_std:.2f}\tTest (Impaired): {test_impaired_mean:.2f}±{test_impaired_std:.2f}')
