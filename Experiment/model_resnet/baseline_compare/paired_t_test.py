import pandas as pd
from scipy.stats import ttest_rel

df1 = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/model_evaluation/summary_model_performance.csv')
df1 = df1[['Full_Name', 'Fold', 'Fold', 'Val_MAE']]
df2 = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation/summary_model_performance.csv')
df2 = df2[['Full_Name', 'Fold', 'Fold', 'Val_MAE']]
df = pd.concat([df1, df2], ignore_index=True)

df_t_statistic = pd.DataFrame(index=df['Full_Name'].unique(), columns=df['Full_Name'].unique())
df_p_value = pd.DataFrame(index=df['Full_Name'].unique(), columns=df['Full_Name'].unique())

for model_1 in df['Full_Name'].unique():
    for model_2 in df['Full_Name'].unique():
        # Perform paired t-test
        val_mae_1 = df.loc[df['Full_Name']==model_1, 'Val_MAE'].values
        val_mae_2 = df.loc[df['Full_Name']==model_2, 'Val_MAE'].values
        
        t_statistic, p_value = ttest_rel(val_mae_1, val_mae_2)
        
        df_t_statistic.loc[model_1, model_2] = t_statistic
        df_p_value.loc[model_1, model_2] = p_value
 
print('t_statistic:')
print(df_t_statistic)
print('p-value:')
print(df_p_value)

df_t_statistic.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/t-statistic.csv')
df_p_value.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/p-value.csv')