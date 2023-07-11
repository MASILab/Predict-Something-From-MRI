# After the manual quality check,
# list out the samples that remains.
# these samples will be used for model training and testing.
# 
# Author: Chenyu Gao
# Date: Jul 11, 2023

from pathlib import Path
import pandas as pd

path_quality_check = Path('/home-local/Projects/PredictAge_QualityCheck/rater-1_offset-0_done')
path_processed_data = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/processed_data')

list_df_subject = []
list_df_session = []
list_df_sample = []
list_df_fa = []
list_df_md = []

for dataset in path_quality_check.iterdir():
    print('Start looping through {}'.format(dataset.name.replace('_done','')))
    
    for fn in dataset.iterdir():
        if not fn.name.endswith('.png'):
            continue
        
        subject = fn.name.split('_')[0]
        session = fn.name.split('_sample')[0]
        sample = fn.name.split('_')[2].replace('.png','')
        
        path_sample = path_processed_data / dataset.name.replace('_done','') / subject / session / sample 
        path_fa = path_sample / 'dwmri%fa_brain_MNI152_linear.nii.gz'
        path_md = path_sample / 'dwmri%md_brain_MNI152_linear.nii.gz'
        
        if not (path_fa.is_file() and path_md.is_file()):
            print('Warning: file not existing', path_sample)
        else:
            list_df_subject.append(subject)
            list_df_session.append(session)
            list_df_sample.append(sample)
            list_df_fa.append(path_fa)
            list_df_md.append(path_md)

d = {'Subject': list_df_subject,
     'Session': list_df_session,
     'Sample': list_df_sample,
     'Path_FA': list_df_fa,
     'Path_MD': list_df_md}

pd.DataFrame(data=d).to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/quality_check/quality_check_results.csv', index=False)