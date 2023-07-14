# 1. Exclude data without demographic/diagnosis info.
# 2. Split healthy subjects into training and testing.
#    Use cognitively impaired subjects as another testing set.
# 
# Author: Chenyu Gao
# Date: Jul 13, 2023

import numpy as np
import pandas as pd

SEED = 42
TRAIN_RATIO = 0.93

# The spreadsheet containing subject information and input image paths
df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/data_all_with_subject_info.csv')

# Drop rows with NaN values
df.dropna(axis=0, inplace=True)

# Training and testing set
list_subjects_healthy_train = []
list_subjects_healthy_test = []
list_subjects_impaired_test = []

df_healthy = df.loc[df['Diagnosis']=='normal']  # note that it's possible some subjects may become impaired subsequently

for dataset in df_healthy['Dataset'].unique():
    
    # randomly sample subjects for each dataset
    subjects = df_healthy.loc[df_healthy['Dataset']==dataset, 'Subject'].unique()
    
    np.random.seed(SEED)
    subjects_train = np.random.choice(subjects, size=round(len(subjects)*TRAIN_RATIO), replace=False)
    subjects_test = subjects[~np.isin(subjects, subjects_train)]
    
    list_subjects_healthy_train += subjects_train.tolist()
    list_subjects_healthy_test += subjects_test.tolist()

list_subjects_impaired_test = df.loc[df['Diagnosis']!='normal', 'Subject'].unique().tolist()

# Check if there is information leakage
for sub in list_subjects_impaired_test:
    if sub in list_subjects_healthy_train:
        print('Notice: potential information leakage.', sub, 'Removing it from training set.')
        list_subjects_healthy_train.remove(sub)
        list_subjects_healthy_test.append(sub)

# Double-check
for subject in list_subjects_impaired_test:
    if subject in list_subjects_healthy_train:
        print('Warning: information leakage!')

print('\n<Summary of train/test split>\n#subjects\nTrain (healthy): {}\n'
      'Test (healthy): {}\nTest (impaired); {}\n'.format(
          len(list_subjects_healthy_train),
          len(list_subjects_healthy_test),
          len(list_subjects_impaired_test)))
print('There is no overlapping subject between Train and Test.\n'
      'There are {} subjects existing in both Test (healthy) and Test (impaired).'
      '\nIn total, there are {} subjects.'.format(
          len(list(set(list_subjects_healthy_test) & set(list_subjects_impaired_test))),
          len(df['Subject'].unique())
          ))

# Save splitting to csv
df[df['Subject'].isin(list_subjects_healthy_train)].to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split/train.csv', index=False)
df[df['Subject'].isin(list_subjects_healthy_test)].to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split/test_healthy.csv', index=False)
df[df['Subject'].isin(list_subjects_impaired_test)].to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split/test_impaired.csv', index=False)