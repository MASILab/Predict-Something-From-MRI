# After running final_testing.py, summarize the model performances.
# Calculate the metrics cross-sectionally.
# 
# Author: Chenyu Gao
# Date: Jul 29, 2023

import pandas as pd
from pathlib import Path
from model import MLP
from sklearn.metrics import mean_squared_error, mean_absolute_error

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(path_csv):
    """Compute the metrics cross-sectionally

    Args:
        path_csv (str or pathlib): path to the csv
    """
    df = pd.read_csv(path_csv)
    
    # Pick only one sample from each subject 
    # at youngest age (for consistency across models in the same fold)
    df_one = df.loc[df.groupby('Subject')['Age'].idxmin()]
    
    mse = mean_squared_error(df_one['Age_gt'], df_one['Age_predicted'])
    mae = mean_absolute_error(df_one['Age_gt'], df_one['Age_predicted'])
    return mse, mae

# Dir of the spreadsheets
path_evaluation_folder = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation')

# Collect the following
list_df_fullname = []  # model's full name
list_df_hidden_layer_sizes = []  # 
list_df_num_params = []  # number of trainable parameters
list_df_fold = []  # which fold was the model trained on

list_df_val_mse = []  # mean squared error
list_df_val_mae = []  # mean absolute error
list_df_test_healthy_mse = []
list_df_test_healthy_mae = []
list_df_test_impaired_mse = []
list_df_test_impaired_mae = []

for fullname in path_evaluation_folder.iterdir():
    if not fullname.name.startswith('MLP-'): continue
    
    # parse the fullname
    hidden_layer_sizes = [int(x) for x in fullname.name.replace('MLP-','').split('-')]
    model = MLP(hidden_layer_sizes=hidden_layer_sizes)
    # model complexity
    num_params = count_parameters(model)

    for fold in fullname.iterdir():
        if not fold.name.startswith('fold-'): continue
        
        list_df_fullname.append(fullname.name)
        list_df_hidden_layer_sizes.append(hidden_layer_sizes)
        list_df_num_params.append(num_params)
        list_df_fold.append(fold.name)
        
        # csv filenames
        csv_val = fold / f"{fullname.name}_{fold.name}_prediction_validation.csv"
        csv_test_healthy = fold / f"{fullname.name}_{fold.name}_prediction_test_healthy.csv"
        csv_test_impaired = fold / f"{fullname.name}_{fold.name}_prediction_test_impaired.csv"
    
        # Validation
        mse, mae = compute_metrics(csv_val)
        list_df_val_mse.append(mse)
        list_df_val_mae.append(mae)

        # Testing (healthy)
        mse, mae = compute_metrics(csv_test_healthy)
        list_df_test_healthy_mse.append(mse)
        list_df_test_healthy_mae.append(mae)
        
        # Testing (impaired)
        mse, mae = compute_metrics(csv_test_impaired)
        list_df_test_impaired_mse.append(mse)
        list_df_test_impaired_mae.append(mae)

# Save to csv
d = {
    'Full_Name': list_df_fullname,
    'Hidden_Layer_Sizes': list_df_hidden_layer_sizes,
    'Num_Params': list_df_num_params,
    'Fold': list_df_fold,
    'Val_MSE': list_df_val_mse,
    'Val_MAE': list_df_val_mae,
    'Test_Healthy_MSE': list_df_test_healthy_mse,
    'Test_Healthy_MAE': list_df_test_healthy_mae,
    'Test_Impaired_MSE': list_df_test_impaired_mse,
    'Test_Impaired_MAE': list_df_test_impaired_mae,
}

df = pd.DataFrame(data=d)
df.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/model_evaluation/summary_model_performance.csv',
          index=False)