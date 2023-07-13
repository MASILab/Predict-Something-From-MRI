# Report a brief summary of the data with subject information
# 
# Author: Chenyu Gao
# Date: Jul 13, 2023

import pandas as pd

df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/data_all_with_subject_info.csv')
print('\nAfter quality check, we have FA + MD images from\n')
for dataset in df['Dataset'].unique():
    print("{}: {} scans from {} sessions from {} subjects".format(dataset,
                                                                    len(df.loc[df['Dataset']==dataset].index),
                                                                    len(df.loc[df['Dataset']==dataset, 'Session'].unique()),
                                                                    len(df.loc[df['Dataset']==dataset, 'Subject'].unique())))

print('\nThe one with age information\n')
df = df.dropna(subset=['Age'], inplace=False)
for dataset in df['Dataset'].unique():
    print("{}: {} scans from {} sessions from {} subjects".format(dataset,
                                                                  len(df.loc[df['Dataset']==dataset].index),
                                                                  len(df.loc[df['Dataset']==dataset, 'Session'].unique()),
                                                                  len(df.loc[df['Dataset']==dataset, 'Subject'].unique())))

print('\nThe one with age/sex/diagnosis information\n')
df.dropna(inplace=True)
for dataset in df['Dataset'].unique():
    print("{}: {} scans from {} ({}* normal / {} cog. impaired) "
          "sessions from {} ({}* normal / {} cog. impaired) subjects".format(
              dataset,
              len(df.loc[df['Dataset']==dataset].index),
              len(df.loc[df['Dataset']==dataset, 'Session'].unique()),
              len(df.loc[(df['Dataset']==dataset)&(df['Diagnosis']=='normal'), 'Session'].unique()),
              len(df.loc[(df['Dataset']==dataset)&(df['Diagnosis']!='normal'), 'Session'].unique()),
              len(df.loc[df['Dataset']==dataset, 'Subject'].unique()),
              len(df.loc[(df['Dataset']==dataset)&(df['Diagnosis']=='normal'), 'Subject'].unique()),
              len(df.loc[(df['Dataset']==dataset)&(df['Diagnosis']!='normal'), 'Subject'].unique())
              )
          )
print("\n*: some subject turned to MCI/AD/impaired (not MCI) in the subsequent sessions.\n")