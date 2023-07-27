import pdb
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt

def get_axial_images(dir, slice_id=86):
    path_fa = Path(dir) / 'dwmri%fa_brain_MNI152_linear.nii.gz'
    path_md = Path(dir) / 'dwmri%md_brain_MNI152_linear.nii.gz'
    
    img_fa = nib.load(path_fa)
    img_md = nib.load(path_md)
    
    data_fa = img_fa.get_fdata()
    data_md = img_md.get_fdata()
    
    aspect = img_fa.header.get_zooms()[1] / img_fa.header.get_zooms()[0]
    
    return (data_fa[:,:,slice_id], data_md[:,:,slice_id], aspect)
    

df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/resnet18_MLP_64/fold-5/resnet18_MLP_64_fold-5_prediction_test_healthy.csv')
df['Difference'] = df['Age_predicted'] - df['Age_gt']

list_df_subset = [
    df.loc[(df['Age']<60)&(df['Difference'].abs()<1),],
    df.loc[(df['Age']<60)&(df['Age_predicted']>60)&(df['Age_predicted']<70),],
    df.loc[(df['Age']<60)&(df['Age_predicted']>70),],
    df.loc[(df['Age']>60)&(df['Age']<70)&(df['Age_predicted']<60)&(df['Difference']<-8),],
    df.loc[(df['Age']>60)&(df['Age']<70)&(df['Difference'].abs()<1),],
    df.loc[(df['Age']>60)&(df['Age']<70)&(df['Age_predicted']>70)&(df['Difference']>8),],
    df.loc[(df['Age']>70)&(df['Age_predicted']<60),],
    df.loc[(df['Age']>70)&(df['Age_predicted']>60)&(df['Age_predicted']<70),],
    df.loc[(df['Age']>70)&(df['Difference'].abs()<0.2),],
]

for seed in tqdm(range(100)):
    
    fig, axes = plt.subplots(ncols=6,nrows=3,figsize=(6.5,4))
    
    for i, df_subset in enumerate(list_df_subset):
        sample = df_subset.sample(random_state=seed)
        dir = sample['dir'].item()
        (fa, md, aspect) = get_axial_images(dir, slice_id=86)
        axes[i//3,i*2%6].imshow(fa.T,
                                cmap='gray', vmin=0, vmax=1,
                                origin='lower',
                                aspect=aspect,
                                interpolation='nearest')
        axes[i//3,(i*2+1)%6].imshow(md.T,
                                    cmap='gray', vmin=0, vmax=0.003,
                                    origin='lower',
                                    aspect=aspect,
                                    interpolation='nearest')
        
        age_gt = round(sample['Age'].item())
        age_predict = round(sample['Age_predicted'].item())
        
        axes[i//3,i*2%6].text(0.01, 0.99,
                              f"chronological: {age_gt}\npredicted: {age_predict}",
                              color='white',
                              fontsize=8, fontfamily='Ubuntu Condensed',
                              transform=axes[i//3,i*2%6].transAxes,
                              verticalalignment='top', 
                              bbox=dict(facecolor='None', alpha=0, edgecolor=None))
    for ax in axes.flatten():
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks
        
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f'/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Figure/fig_qualitative_good_bad_predictions/figs/seed_{seed}.png',
                dpi=600,
                bbox_inches='tight')
    plt.close('all')
