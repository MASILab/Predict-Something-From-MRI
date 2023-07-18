# 1. Copy data to local disk.
# 2. Prepare dataframes containing path and age information.
# 
# Author: Chenyu Gao
# Date: Jul 18, 2023

import subprocess
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# Local disk location
path_local_root = Path('/home-local/Projects/Predict_Age_Data')

# List of files to rsync
list_file = ['dwmri%fa_brain_MNI152_linear.nii.gz',
             'dwmri%md_brain_MNI152_linear.nii.gz']

# Dataset information
path_data_split_folder = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split')
list_dataset_csv = [fn for fn in path_data_split_folder.iterdir() if fn.name.endswith('.csv')]

# rsync each data and record updated info to csv
for csv in list_dataset_csv:
    data_category = csv.name.replace('.csv', '')  # "train", "test_healthy", "test_impaired"
    print('rsync {} in progress...'.format(data_category))
    
    df = pd.read_csv(csv)  # Dataset,Subject,Session,Sample,Path_FA,Path_MD,Age,Sex,Diagnosis
    df_move = df.copy(deep=True)
    df_move.drop(columns=['Path_FA', 'Path_MD'], inplace=True)
    df_move['dir'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df.index)):
        
        original_dir = Path(row['Path_FA']).parent
        target_dir = path_local_root / data_category / row['Dataset'] / row['Subject'] / row['Session'] / row['Sample']
        subprocess.run(['mkdir', '-p', target_dir])
        
        for f in list_file:
            source_file = original_dir / f
            subprocess.run(['rsync', '-a', source_file, (str(target_dir)+'/')])
        
        df_move.loc[idx, 'dir'] = target_dir
    
    df_move.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/{}'.format(csv.name))