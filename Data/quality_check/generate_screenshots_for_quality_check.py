# Generate screenshots of the preprocessed DTI scalar images
# for quality check. 
# Save pictures to local disk.
# 
# Author: Chenyu Gao
# Date: Jul 10, 2023

import subprocess
import multiprocessing
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Define each plotting job
def plot_sample(input_tuple):
    
    # Unpack the input tuple
    list_img, path_output_folder, session, sample = input_tuple
    subprocess.run(['mkdir', '-p', path_output_folder])

    # Draw axial, coronal, and sagittal views for each DTI scalar map
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    for idx_col, img in enumerate(list_img):
    
        # Load image
        dti_img = nib.load(img)
        dti_data = np.squeeze(dti_img.get_fdata())
        resolution = dti_data.shape[:3] 
        # Take screenshots  
        # axial
        axes[0,idx_col].set_title(img.name.split('%')[1].split('_')[0], fontsize=25)
        aspect = dti_img.header.get_zooms()[1] / dti_img.header.get_zooms()[0]
        axes[0,idx_col].imshow(dti_data[:,:,round(resolution[2]/2)+offset].T,
                               cmap='gray', 
                               origin='lower',
                               aspect=aspect,
                               interpolation='nearest')
        # coronal
        aspect = dti_img.header.get_zooms()[2] / dti_img.header.get_zooms()[0]
        axes[1,idx_col].imshow(dti_data[:,round(resolution[1]/2)+offset,:].T,
                               cmap='gray', 
                               origin='lower',
                               aspect=aspect,
                               interpolation='nearest')
        # sagittal
        aspect = dti_img.header.get_zooms()[2] / dti_img.header.get_zooms()[1]
        axes[2,idx_col].imshow(dti_data[round(resolution[0]/2)+offset+5,:,:].T,
                               cmap='gray', 
                               origin='lower',
                               aspect=aspect,
                               interpolation='nearest')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        fig_save = path_output_folder / "{}_{}.png".format(session, sample)
        fig.savefig(fig_save, bbox_inches='tight')
        plt.close('all')


# Prepare list of tuples of inputs for parallel processing [(*,*,*,*),(*,*,*,*),(*,*,*,*)...]
path_datasets = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/processed_data')
path_quality_check = Path('/home-local/Projects/PredictAge_QualityCheck')
rater_offsets = [0, -20, 20]  # how much deviation from the center slice

list_input_tuple = []

for i,offset in enumerate(rater_offsets):
    for dataset in path_datasets.iterdir():
        if not dataset.is_dir(): continue

        # Screenshot output directory
        path_output_folder = path_quality_check / "rater-{}_offset-{}".format(str(i+1), offset) / dataset.name
        
        for subject in dataset.iterdir():
            if not subject.name.startswith('sub-'): continue
            
            for session in subject.iterdir():
                if not ('ses' in session.name) and session.is_dir(): continue
                
                for sample in session.iterdir():
                    if not sample.name.startswith('sample-'): continue
                    
                    path_fa_img = sample / "dwmri%fa_brain_MNI152_linear.nii.gz"
                    path_md_img = sample / "dwmri%md_brain_MNI152_linear.nii.gz"
                    path_ad_img = sample / "dwmri%ad_brain_MNI152_linear.nii.gz"
                    path_rd_img = sample / "dwmri%rd_brain_MNI152_linear.nii.gz"
                    
                    list_img = [path_fa_img, path_md_img, path_ad_img, path_rd_img]
                    
                    list_input_tuple.append((list_img, path_output_folder, session.name, sample.name))


# Parallel processing
pool = multiprocessing.Pool(processes=10)
pool.map(plot_sample, tqdm(list_input_tuple), chunksize=1)

pool.close()
pool.join()
