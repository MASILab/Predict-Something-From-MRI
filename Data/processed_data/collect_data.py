# Symlink data with selected bval=[0, 700] for SPIE project
# 
# Author: Chenyu Gao
# Date: Jul 7, 2023

import subprocess
import pandas as pd
from pathlib import Path

# Where to save (symlink to) data
path_dataset_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/processed_data')

# Load dataframe of selected samples
df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/search_similar_bval/bval_search_selected.csv')

# Create folder symlink files over for each sample
for _, row in df.iterrows():
    dataset, subject, session, path_prequal = row[['Dataset', 'Subject', 'Session', 'Path_PreQual']]
    
    path_wmatlas_folder = Path(path_prequal).parent / path_prequal.split('/')[-1].replace('PreQual', 'WMAtlas')
    
    # collect necessary (and some optional) data
    path_fa_img = path_wmatlas_folder / "dwmri%fa.nii.gz"
    path_md_img = path_wmatlas_folder / "dwmri%md.nii.gz"
    path_ad_img = path_wmatlas_folder / "dwmri%ad.nii.gz"
    path_rd_img = path_wmatlas_folder / "dwmri%rd.nii.gz"
    path_slant_ticv_img = path_wmatlas_folder / "dwmri%T1_seg_to_dwi.nii.gz"
    path_transform_b0_to_t1 = path_wmatlas_folder / "dwmri%ANTS_b0tot1.txt"
    path_transform_t1_to_MNI_affine = path_wmatlas_folder / "dwmri%0GenericAffine.mat"
    path_transform_t1_to_MNI_warp = path_wmatlas_folder / "dwmri%1Warp.nii.gz"
    
    # Check if any of these is missing
    MISSING = False
    list_files = [path_fa_img, path_md_img, path_ad_img, path_rd_img, 
                  path_slant_ticv_img, 
                  path_transform_b0_to_t1, 
                  path_transform_t1_to_MNI_affine, path_transform_t1_to_MNI_warp]
    for fn in list_files:
        if not fn.is_file():
            MISSING = True
            break
    if MISSING:
        print("Notice: {} has missing file(s). Skip.".format(path_wmatlas_folder))
        continue

    # mkdir for destination folder
    path_dataset_session = path_dataset_root / dataset / subject / session
    subprocess.run(['mkdir', '-p', path_dataset_session])
    
    num_existing = len([fn for fn in path_dataset_session.iterdir() if fn.name.startswith('sample-')])
    path_sample_folder = path_dataset_session / "sample-{:02}".format(num_existing+1)
    subprocess.run(['mkdir', '-p', path_sample_folder])
    
    # symlink files over
    for fn in list_files:
        source_file =  fn
        target_link = path_sample_folder / fn.name
        try:
            subprocess.run(['ln', '-sf', source_file, target_link])
        except:
            print('Warning: symlink failure {}'.format(source_file))
