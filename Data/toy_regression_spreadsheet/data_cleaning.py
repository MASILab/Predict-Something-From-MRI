# Prepare and clean the data for the toy experiment (predict age from values extracted from MRI).
# 
# Author: Chenyu Gao
# Date: Jun 29, 2023

import pandas as pd
from tqdm import tqdm

path_csv_brain_stats = '/home/local/VANDERBILT/gaoc11/Projects/Variance-Aging-Diffusion/Data/BLSA_Brain_stats_concat_20221110_delivery.csv'
path_csv_motion = '/home/local/VANDERBILT/gaoc11/Projects/Variance-Aging-Diffusion/Data/BLSA_eddy_movement_rms_average_20221109.csv'
csv_save = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/toy_regression_spreadsheet/toy_dataset.csv'

# Load measurements data
df_brain_stats = pd.read_csv(path_csv_brain_stats)

# Drop some columns (otherwise there will be too many features!)
list_col2drop = []
for col in df_brain_stats.columns:
    if col == 'Session':
        continue
    if not 'DTI1' in col:
        list_col2drop.append(col)
    if not 'EveType1' in col:
        list_col2drop.append(col)
    if 'std' in col:
        list_col2drop.append(col)
df_brain_stats.drop(list_col2drop, axis=1, inplace=True)

# Append age and sex information
df_motion = pd.read_csv(path_csv_motion)  # motion spreadsheet containing "age" and "sex" columns

df_brain_stats['Age'] = None
df_brain_stats['Sex'] = None

for _,row in df_brain_stats.iterrows():
    age = df_motion.loc[(df_motion['Session']==row['Session']) & (df_motion['DTI_ID']==1), 'Age'].values
    if len(age)==0:
        print('Notice: cannot find age info for session {}'.format(row['Session']))
    else:
        df_brain_stats.loc[df_brain_stats['Session']==row['Session'], 'Age'] = age[0]
    
    sex = df_motion.loc[(df_motion['Session']==row['Session']) & (df_motion['DTI_ID']==1), 'Sex'].values
    if len(sex)==0:
        print('Notice: cannot find sex info for session {}'.format(row['Session']))
    else:
        df_brain_stats.loc[df_brain_stats['Session']==row['Session'], 'Sex'] = sex[0]

# Drop NaN rows
df_brain_stats.dropna(inplace=True)

df_brain_stats.to_csv(csv_save, index=False)
print(df_brain_stats)
