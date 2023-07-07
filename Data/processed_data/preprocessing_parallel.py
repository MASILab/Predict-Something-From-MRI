import subprocess
import multiprocessing
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# Define processings for each folder
def preprocess_sample_folder(path_sample_folder):
    
    # Required files
    path_t1_MNI = '/nfs2/ForChenyu/MNI_152.nii.gz'  # MNI template T1 image
    path_slant_ticv_img = Path(path_sample_folder) / 'dwmri%T1_seg_to_dwi.nii.gz'
    path_transform_b0_to_t1 = Path(path_sample_folder) / 'dwmri%ANTS_b0tot1.txt'
    path_transform_t1_to_MNI_affine = Path(path_sample_folder) / 'dwmri%0GenericAffine.mat'
    path_transform_t1_to_MNI_warp = Path(path_sample_folder) / 'dwmri%1Warp.nii.gz'
    
    # Preprocess each DTI scalar map
    list_dti_scalar = ['fa', 'md', 'rd', 'ad']
    for dti_scalar in list_dti_scalar:

        path_dti_scalar_img = Path(path_sample_folder) / "dwmri%{}.nii.gz".format(dti_scalar)
        
        # Step 1: brain extraction using SLANT
        dti_scalar_img = nib.load(path_dti_scalar_img)
        slant_ticv_img = nib.load(path_slant_ticv_img)

        dti_scalar_data = dti_scalar_img.get_fdata()
        slant_ticv_data = slant_ticv_img.get_fdata()

        dti_scalar_brain_data = np.where(slant_ticv_data == 0, 0, dti_scalar_data)
        dti_scalar_brain_img = nib.Nifti1Image(dti_scalar_brain_data, dti_scalar_img.affine, header=dti_scalar_img.header)
        path_dti_scalar_brain_img = Path(path_sample_folder) / "dwmri%{}_brain.nii.gz".format(dti_scalar)
        nib.save(dti_scalar_brain_img, path_dti_scalar_brain_img)
    
        # Step 2: align to MNI space    
        output = Path(path_sample_folder) / "dwmri%{}_brain_MNI152_linear.nii.gz".format(dti_scalar)
        subprocess.run(['antsApplyTransforms',
                        '-d', '3',
                        '-i', path_dti_scalar_brain_img,
                        '-r', path_t1_MNI,
                        '-o', output,
                        '-n', 'Linear',
                        '-t', path_transform_b0_to_t1,
                        '-t', path_transform_t1_to_MNI_affine,
                        '-t', path_transform_t1_to_MNI_warp
                        ])


# Prepare a list of paths to sample-* folders for parallel processing
path_processed_data = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/processed_data')
list_sample_folder = []

for dataset in path_processed_data.iterdir():
    if not dataset.is_dir():
        continue
    for subject in dataset.iterdir():
        for session in subject.iterdir():
            for sample in session.iterdir():
                list_sample_folder.append(sample)


# Parallel processing
pool = multiprocessing.Pool(processes=20)
pool.map(preprocess_sample_folder, tqdm(list_sample_folder), chunksize=1)

pool.close()
pool.join()
