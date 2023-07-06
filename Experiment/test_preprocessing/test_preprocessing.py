# Test the code for pre-processing, 
# which involves brain extraction and alignment to MNI space.
# Author: Chenyu Gao
# Date: Jul 5, 2023

import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path

path_test_data = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/test_preprocessing/sample_data')
path_t1_MNI = '/nfs2/ForChenyu/MNI_152.nii.gz'
path_output_dir = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/test_preprocessing/')

for subject in path_test_data.iterdir():
    
    # Step 1: brain extraction using SLANT
    path_fa_img = subject / "dwmri%fa.nii.gz"
    path_slant_ticv_img = subject / 'dwmri%T1_seg_to_dwi.nii.gz'
    
    fa_img = nib.load(path_fa_img)
    slant_ticv_img = nib.load(path_slant_ticv_img)
    
    fa_data = fa_img.get_fdata()
    slant_ticv_data = slant_ticv_img.get_fdata()

    fa_data_masked = np.where(slant_ticv_data == 0, 0, fa_data)
    masked_fa_img = nib.Nifti1Image(fa_data_masked, fa_img.affine, header=fa_img.header)
    path_masked_fa_img = subject / "dwmri%fa_brain.nii.gz"
    nib.save(masked_fa_img, path_masked_fa_img)
    
    # Step 2: align to MNI space
    t_b0_2_t1 = subject / 'dwmri%ANTS_b0tot1.txt'
    t_t1_2_MNI_affine = subject / 'dwmri%0GenericAffine.mat'
    t_t1_2_MNI_warp = subject / 'dwmri%1Warp.nii.gz'
    
    for interpolation in ['Linear']:
        output = path_output_dir / '{}_fa_brain_MNI_{}.nii.gz'.format(subject.name, interpolation)        

        subprocess.run(['antsApplyTransforms',
                        '-d', '3',
                        '-i', path_masked_fa_img,
                        '-r', path_t1_MNI,
                        '-o', output,
                        '-n', interpolation,
                        '-t', t_b0_2_t1,
                        '-t', t_t1_2_MNI_affine,
                        '-t', t_t1_2_MNI_warp
                        ])

