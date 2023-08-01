import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df_healthy = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/resnet18_noMLP/fold-5/resnet18_noMLP_fold-5_prediction_test_healthy.csv')
# df_impaired = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/resnet18_noMLP/fold-5/resnet18_noMLP_fold-5_prediction_test_impaired.csv')

df_healthy = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/resnet18_MLP_64/fold-5/resnet18_MLP_64_fold-5_prediction_test_healthy.csv')
df_impaired = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/resnet18_MLP_64/fold-5/resnet18_MLP_64_fold-5_prediction_test_impaired.csv')
df1 = pd.concat([df_healthy, df_impaired], ignore_index=True)
df1 = df1[['Subject','Session','Sample','Age','Diagnosis','Age_gt','Age_predicted']]
df1['Method'] = '3DResNet'

df_healthy = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation/MLP-128-64/fold-5/MLP-128-64_fold-5_prediction_test_healthy.csv')
df_impaired = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation/MLP-128-64/fold-5/MLP-128-64_fold-5_prediction_test_impaired.csv')
df2 = pd.concat([df_healthy, df_impaired], ignore_index=True)
df2 = df2[['Subject','Session','Sample','Age','Diagnosis','Age_gt','Age_predicted']]
df2['Method'] = 'ROIBasedFeature'

df = pd.concat([df1, df2], ignore_index=True)

# Sanity check of the data
test = df['Age'] - df['Age_gt'] 
for i in range(len(df.index)):
    # 'Age' is the original label. 'Age_gt' is retrieved during model inferencing. They should have same value (except for datatype)
    if test[i] >= 0.01:
        print(i, test[i])

# Bland–Altman plot
df['Difference'] = df['Age_predicted'] - df['Age_gt']

dict_rename = {'normal':'Normal', 
               'impaired (not MCI)':'Impaired (not MCI)',
               'MCI': 'MCI',
               'dementia':'Dementia'}

for i,method in enumerate(df['Method'].unique()):
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(6.5, 1.5), sharex=True, sharey=True)

    for j, diagnosis in enumerate(['normal', 'impaired (not MCI)', 'MCI', 'dementia']):
        data = df.loc[(df['Method']==method)&(df['Diagnosis']==diagnosis), ]
        data = data.loc[data.groupby('Subject')['Age'].idxmin()]  # pick the first sample of each subject (cross-sectional analysis)

        sns.scatterplot(data=data,
                        x='Age_gt',
                        y='Difference',
                        s=6,
                        color="tab:blue",
                        linewidth=0,
                        ax=axes[j])
        sns.kdeplot(data=data,
                    x='Age_gt',
                    y='Difference', 
                    color='tab:blue', 
                    fill=True, 
                    levels= 10, 
                    cut=1,
                    alpha=.9,
                    ax=axes[j])
        axes[j].axhline(y=0, linestyle='--',linewidth=1, color='k', alpha=0.25)
        
        axes[j].set_xlabel('')
        axes[j].set_ylabel('')
        
        axes[j].text(0.05, 0.95, 
                    f"{dict_rename[diagnosis]}", 
                    fontsize=9,fontfamily='Ubuntu Condensed',
                    transform=axes[j].transAxes,
                    verticalalignment='top', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

        axes[j].tick_params(axis='both', direction='out', length=2)
        axes[j].set_xlim(25, 105)
        axes[j].set_ylim(-25, 25)
        axes[j].set_xticks([35, 65, 95])
        axes[j].set_yticks([-15,0,15])

    fig.subplots_adjust(hspace=0,wspace=0.03)
    fig.savefig(f'/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Figure/fig_Bland_Altman/v3/bland_altman_v3_{method}.png',
                dpi=600,
                bbox_inches='tight')
