# Prepare spreadsheets (like the one Qi did for BLSA, but smaller) 
# of ROI-based measures of FA and MD.
# The spreadsheet will be used for training and testing of the 
# baseline models (MLP, RandomForest etc).
# 
# Author: Chenyu Gao
# Date: Jul 14, 2023

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
from multiprocessing import Pool

def process_row(row_tuple):

    global roi_id_full
    
    Dataset, Subject, Session, Sample, Path_FA, Path_MD, Age, Sex, Diagnosis = row_tuple
    
    # SLANT_TICV segmentation and FA image
    path_t1_seg = Path(Path_FA).parent / 'dwmri%T1_seg_to_dwi.nii.gz'
    path_fa = Path(Path_FA).parent / 'dwmri%fa.nii.gz'
    path_md = Path(Path_FA).parent / 'dwmri%md.nii.gz'
    
    img_t1_seg = nib.load(path_t1_seg)
    img_fa = nib.load(path_fa)
    img_md = nib.load(path_md)
    
    data_t1_seg = img_t1_seg.get_fdata()
    data_fa = img_fa.get_fdata()
    data_md = img_md.get_fdata()
    
    # Loop through ROIs and extract measure
    measure = {'Dataset':Dataset, 'Subject':Subject, 'Session':Session, 
               'Sample':Sample, 'Age':Age, 'Sex':Sex, 'Diagnosis':Diagnosis}
    
    for id in roi_id_full:
        
        indices = np.asarray(data_t1_seg == id).nonzero() # selected roi
        
        if np.sum(indices) == 0:
            measure['ROI-{}_FA_mean'.format(id)] = None
            measure['ROI-{}_FA_std'.format(id)] = None
            measure['ROI-{}_MD_mean'.format(id)] = None
            measure['ROI-{}_MD_std'.format(id)] = None
        else:
            measure['ROI-{}_FA_mean'.format(id)] = np.nanmean(data_fa[indices])
            measure['ROI-{}_FA_std'.format(id)] = np.nanstd(data_fa[indices])
            measure['ROI-{}_MD_mean'.format(id)] = np.nanmean(data_md[indices])
            measure['ROI-{}_MD_std'.format(id)] = np.nanstd(data_md[indices])
    
    return measure

# Main
# csv files containing dataset information
list_dataset_csv = ['/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split/train.csv',
                    '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split/test_healthy.csv',
                    '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/train_test_split/test_impaired.csv']

# SLANT look up table (ROI_ID we want to loop through)
lut = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_baseline_ROI-based-measure/LUT.csv')
roi_id_full = lut['id'].values

for path_csv in list_dataset_csv:
    
    dataset_class = path_csv.split('/')[-1].replace('.csv','')
    print('Extracting measures for', dataset_class)
    
    # Load the csv containing the file path and subject info
    df = pd.read_csv(path_csv)
    
    # Process rows of the dataframe in parallel
    rows =  [tuple(x) for x in df.values]
    with Pool(processes=15) as pool:
        results = list(tqdm(pool.imap(process_row, rows, chunksize=1), total=df.shape[0]))
    
    # Concatenate the results with the original DataFrame
    result_df = pd.DataFrame.from_records(results)
    
    # Save to csv
    saveto = '/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_baseline_ROI-based-measure/data/ROIbased_measure_{}.csv'.format(dataset_class)
    result_df.to_csv(saveto, index=False)
    