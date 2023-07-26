# Summarize the dataset information for writing up the paper.
# 
# Author: Chenyu Gao
# Date: Jul 26, 2023

import pandas as pd
import numpy as np

csv_train = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/train.csv'
csv_test_healthy = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/test_healthy.csv'
csv_test_impaired = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/test_impaired.csv'

dict_dataset = {
    'Train/Val': csv_train,
    'Test_healthy': csv_test_healthy,
    'Test_impaired': csv_test_impaired,
}

for dataset in dict_dataset.keys():
    print(dataset)
    df = pd.read_csv(dict_dataset[dataset])
    
    for site in df['Dataset'].unique():
    
        num_subjects = len(df.loc[df['Dataset']==site, 'Subject'].unique())
        
        unique_subject_ages = []
        for sub in df.loc[df['Dataset']==site, 'Subject'].unique():
            age = df.loc[(df['Dataset']==site)&(df['Subject']==sub), 'Age'].min()
            unique_subject_ages.append(age)
        
        age_mean = np.nanmean(unique_subject_ages) 
        age_std = np.nanstd(unique_subject_ages)
        
        num_young = 0
        num_middle = 0
        num_old = 0
        
        for age in unique_subject_ages:
            if age <= 50:
                num_young += 1
            elif (age > 50) & (age <=70):
                num_middle += 1
            elif age > 70:
                num_old += 1
            else:
                print("Error!")
        
        print(f"{site}\t#Subjects: {num_subjects}\tMean Age: {age_mean:.1f}±{age_std:.1f}\t#0-50 y/o: {num_young}\t#50-70 y/o: {num_middle}\t#>=70 y/o: {num_old}")
        
    # Total
    num_subjects = len(df['Subject'].unique())
    unique_subject_ages = []
    for sub in df['Subject'].unique():
        age = df.loc[df['Subject']==sub, 'Age'].min()
        unique_subject_ages.append(age)
        
    age_mean = np.nanmean(unique_subject_ages) 
    age_std = np.nanstd(unique_subject_ages)
    
    num_young = 0
    num_middle = 0
    num_old = 0
    
    for age in unique_subject_ages:
        if age <= 50:
            num_young += 1
        elif (age > 50) & (age <=70):
            num_middle += 1
        elif age > 70:
            num_old += 1
        else:
            print("Error!")
    
    print(f"Total\t#Subjects: {num_subjects}\tMean Age: {age_mean:.1f}±{age_std:.1f}\t#0-50 y/o: {num_young}\t#50-70 y/o: {num_middle}\t#>=70 y/o: {num_old}")