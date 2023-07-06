# Search through the selected datasets (in BIDS-format) and collect bval from PreQual folders.
# After this, we will decide which bval to filter data for building and testing the model.
# 
# Author: Chenyu Gao
# Date: Jul 4, 2023

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pdb

# Path of the datasest in BIDS
path_bids = Path('/nfs2/harmonization/BIDS')
list_datasets_prequal = ['BIOCARD', 'BLSA', 'ICBM', 'UKBB', 'ABVIB', 'VMAP', 'OASIS3', 'ADNI']

# Collect the following information
list_df_dataset = []
list_df_subject = []
list_df_session = []
list_df_prequal = []
list_df_bval = []
list_df_bval_unique = []

for dataset in list_datasets_prequal:
    print('Start searching through {}...'.format(dataset))
    
    list_subject = [fn for fn in (path_bids/dataset/'derivatives').iterdir() if (fn.name.startswith('sub-') and fn.is_dir())]
    for subject in tqdm(list_subject):
        
        list_session = [fn for fn in subject.iterdir() if (fn.name.startswith('ses-') and fn.is_dir())]
        if len(list_session)>0:
            for session in list_session:
                
                list_prequal = [fn for fn in session.iterdir() if (fn.name.startswith('PreQual') and fn.is_dir())]
                for prequal in list_prequal:
                    if (prequal / 'OPTIMIZED_BVECS' / 'dwmri.bval').is_file():
                        bval = np.loadtxt((prequal / 'OPTIMIZED_BVECS' / 'dwmri.bval'))
                        bval_round = np.sort(np.round(bval))
                        bval_round_unique = np.unique(bval_round)
                        
                        list_df_dataset.append(dataset)
                        list_df_subject.append(subject.name)
                        list_df_session.append("{}_{}".format(subject.name,session.name))
                        list_df_prequal.append(prequal)
                        list_df_bval.append(bval_round.tolist())
                        list_df_bval_unique.append(bval_round_unique.tolist())
        else:
            list_prequal = [fn for fn in subject.iterdir() if (fn.name.startswith('PreQual') and fn.is_dir())]
            for prequal in list_prequal:
                if (prequal / 'OPTIMIZED_BVECS' / 'dwmri.bval').is_file():
                    bval = np.loadtxt((prequal / 'OPTIMIZED_BVECS' / 'dwmri.bval'))
                    bval_round = np.sort(np.round(bval))
                    bval_round_unique = np.unique(bval_round)
                    
                    list_df_dataset.append(dataset)
                    list_df_subject.append(subject.name)
                    list_df_session.append("{}_ses-1".format(subject.name))
                    list_df_prequal.append(prequal)
                    list_df_bval.append(bval_round.tolist())
                    list_df_bval_unique.append(bval_round_unique.tolist())

d = {'Dataset': list_df_dataset,
     'Subject': list_df_subject,
     'Session': list_df_session,
     'Path_PreQual': list_df_prequal,
     'bval': list_df_bval,
     'bval_unique': list_df_bval_unique}
df = pd.DataFrame(data=d)
# df.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/search_similar_bval/bval_search_all.csv', index=False)

# Count the number of samples acquired with the same bval
list_type = []
for bval_unique in list_df_bval_unique:
    if bval_unique not in list_type:
        list_type.append(bval_unique)
        
        selected_rows = df[df['bval_unique'].apply(lambda x: x == bval_unique)]
        print('\nUnique bvalues: {}\n <Number of sessions using combination of these bvals>'.format(bval_unique))
        for dataset in selected_rows['Dataset'].unique():
            print('{}: {}'.format(dataset, len(selected_rows.loc[selected_rows['Dataset']==dataset, 'Session'].unique())))

# Save the selected bval samples into csv
selected_bval = [0, 700]
df_selected = df[df['bval_unique'].apply(lambda x: x == selected_bval)]
df_selected.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/search_similar_bval/bval_search_selected.csv', index=False)

# pdb.set_trace()