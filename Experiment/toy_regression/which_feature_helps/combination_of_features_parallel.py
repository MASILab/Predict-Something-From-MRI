from functions import *
from tqdm import tqdm
import numpy as np
import multiprocessing
import warnings

random_seed = 0
warnings.filterwarnings('ignore')

# Define the job
def process_row(row_tuple):
    index, row = row_tuple
    atlas, DTI_scalar, value, volume = row[['Atlas', 'DTI_scalar', 'Value', 'IncludeVolumeData']]
    print("Start playing with feature combination: {} {} {} {}".format(atlas, DTI_scalar, value, volume))

    num_input_feat, rf_mae_mean, mlp_mae_mean = age_prediction(atlas, DTI_scalar, value, volume, random_seed=random_seed)
    
    return index, num_input_feat, rf_mae_mean, mlp_mae_mean

# Define combinations to test
atlas_choices = [['EveType1'], ['EveType2'], ['EveType3'], ['BrainColor']]
DTI_scalar_choices = [['FA'], ['MD'], ['RD'], ['AD'], 
                      ['FA', 'MD'], ['FA', 'RD'], ['FA', 'AD'], ['MD', 'RD'], ['MD', 'AD'], ['RD', 'AD'],
                      ['FA', 'MD', 'RD'], ['FA', 'MD', 'AD'], ['FA', 'RD', 'AD'], ['MD', 'RD', 'AD'],
                      ['FA', 'MD', 'RD', 'AD']]
value_choices = [['mean'], ['std'], ['mean', 'std']]
volume_choices = [False, True]

# prepare a dataframe for parallel processing
list_df_atlas = []
list_df_dti_scalar = []
list_df_value = []
list_df_volume = []

for volume in volume_choices:
    for atlas in atlas_choices:
        for DTI_scalar in DTI_scalar_choices:
            for value in value_choices:
                list_df_atlas.append(atlas)
                list_df_dti_scalar.append(DTI_scalar)
                list_df_value.append(value)
                list_df_volume.append(volume)

d = {'Atlas': list_df_atlas,
     'DTI_scalar': list_df_dti_scalar,
     'Value': list_df_value,
     'IncludeVolumeData': list_df_volume}

record = pd.DataFrame(data=d)
record['#InputFeatures'] = 0
record['MAE(RandomForest)'] = 999
record['MAE(MLP)'] = 999

# Parallel processing
pool = multiprocessing.Pool(processes=20)
results = pool.map(process_row, record.iterrows(), chunksize=1)

for index, num_input_feat, rf_mae_mean, mlp_mae_mean in results:
    record.loc[index, '#InputFeatures'] = num_input_feat
    record.loc[index, 'MAE(RandomForest)'] = rf_mae_mean
    record.loc[index, 'MAE(MLP)'] = mlp_mae_mean

pool.close()
pool.join()

# Save results to csv
record.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/toy_regression/which_feature_helps/MAE_different_input_feature_combo.csv', index=False)