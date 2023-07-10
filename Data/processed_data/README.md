# Preprocessing for SPIE23 project

### What's in this folder?

Recall that we decided to use DTI scalar maps from DWI data acquired at bval=700. 
After symlinking data to folders under this project `Predict-Something-From-MRI/Data/processed_data`,
we need to perform preprocessing, which involves two steps:

- Skull stripping.
- Registration to MNI152 template space.

We don't need to run these processing from scratch. 
Instead, Michael Kim ran WhiteMatter Atlas pipeline before, which generates everything we need at present.
The WhiteMatter Atlas project can be found on GitHub at `https://github.com/MASILab/AtlasToDiffusionReg/blob/main/README.md`.

### The files we need are:

- dwmri%fa.nii.gz: FA scalar map image
- dwmri%md.nii.gz: MD scalar map image
- dwmri%ad.nii.gz: AD scalar map image
- dwmri%rd.nii.gz: RD scalar map image
- dwmri%T1_seg_to_dwi.nii.gz: SLANT_TICV segmentation image in b0 space
- dwmri%ANTS_b0tot1.txt: transformation from b0 space to T1 space
- dwmri%0GenericAffine.mat: affine component of the transformation from subject's T1 to MNI152 T1 space
- dwmri%1Warp.nii.gz: non-linear component of the transformation from subject's T1 to MNI152 T1 space

### Test and sanity check

I test the preprocessing code in `Predict-Something-From-MRI/Experiment/test_preprocessing`, with two examples. Note that I also tried BSpline for interpolation but eventually chose Linear.

