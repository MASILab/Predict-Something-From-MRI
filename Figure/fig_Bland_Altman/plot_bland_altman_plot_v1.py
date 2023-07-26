import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_healthy = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/prediction_resnet18_MLP_64/prediction_test_healthy.csv')
df_impaired = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/prediction_resnet18_MLP_64/prediction_test_impaired.csv')
df = pd.concat([df_healthy, df_impaired], ignore_index=True)

# Sanity check of the data
test = df['Age'] - df['Age_gt'] 
for i in range(len(df.index)):
    # 'Age' is the original label. 'Age_gt' is retrieved during model inferencing. They should have same value (except for datatype)
    if test[i] >= 0.01:
        print(i, test[i])

# Bland–Altman plot
df['Difference'] = df['Age_predicted'] - df['Age_gt']
# df['Average'] = (df['Age_predicted'] + df['Age_gt']) / 2

dict_rename = {'normal':'Normal', 
               'impaired (not MCI)':'Impaired (not MCI)',
               'MCI': 'MCI',
               'dementia':'Dementia'}

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5, 6.5), sharex=True, sharey=True)

for i, diagnosis in enumerate(['normal', 'impaired (not MCI)', 'MCI', 'dementia']):

    data = df.loc[df['Diagnosis']==diagnosis,]
    sns.scatterplot(data=data,
                    x='Age_gt',
                    y='Difference',
                    s=10,
                    color="tab:blue",
                    linewidth=0,
                    ax=axes[i//2, i%2])
    sns.kdeplot(data=data,
                x='Age_gt',
                y='Difference', 
                color='tab:blue', 
                fill=True, 
                levels= 10, 
                cut=1,
                alpha=.95,
                ax=axes[i//2, i%2])

    axes[i//2, i%2].set_xlabel('Chronological age (year)')
    axes[i//2, i%2].set_ylabel('Predicted - chronological age (year)')
    
    axes[i//2, i%2].text(0.04, 0.96, 
                         f"{dict_rename[diagnosis]}", 
                         transform=axes[i//2, i%2].transAxes,
                         verticalalignment='top', 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
fig.subplots_adjust(hspace=0.15,wspace=0.15)
fig.savefig('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Figure/fig_Bland_Altman/bland_altman_v1.png',
            dpi=600,
            bbox_inches='tight')
