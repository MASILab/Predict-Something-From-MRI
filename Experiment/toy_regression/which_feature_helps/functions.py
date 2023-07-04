import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import numpy as np


def prepare_dataframe(atlas=['EveType1'], 
                      DTI_scalar=['FA','MD','RD','AD'], 
                      value=['mean','std'],
                      volume=False):
    """Prepare the dataframe which contains the training and testing set 
    for the toy regression experiments. The columns of the dataframe are 
    selected by the user.

    Args:
        atlas (list, optional): selected atlas. Defaults to ['EveType1']. ('EveType1','EveType2','EveType3','BrainColor')
        DTI_scalar (list, optional): selected DTI scalars. Defaults to ['FA','MD','RD','AD'].
        value (list, optional): selected summary values. Defaults to ['mean','std'].
        volume (bool, optional): whether to include volumes. Defaults to False.

    Returns:
        df_brain_stats: pandas dataframe
    """
    
    df_brain_stats = pd.read_csv('/home/local/VANDERBILT/gaoc11/Projects/Variance-Aging-Diffusion/Data/BLSA_Brain_stats_concat_20221110_delivery.csv')
    df_motion = pd.read_csv('/home/local/VANDERBILT/gaoc11/Projects/Variance-Aging-Diffusion/Data/BLSA_eddy_movement_rms_average_20221109.csv')  # motion spreadsheet containing "age" and "sex" columns

    # Drop unselected columns
    list_col2drop = []
    for col in df_brain_stats.columns:
        if col == 'Session':
            continue
        
        if not col.split('-')[0] in atlas:
            list_col2drop.append(col)
            continue
        
        if ('DTI2' in col) or ('DTI_double' in col):
            list_col2drop.append(col)
            continue
        
        if 'Volume' in col:
            if volume==False:
                list_col2drop.append(col)
                continue
        else:
            if (col.split('-')[-2] not in DTI_scalar) or (col.split('-')[-1] not in value):
                list_col2drop.append(col)
                continue
    df_brain_stats.drop(list_col2drop, axis=1, inplace=True)

    # Append age and sex info
    df_brain_stats['Age'] = None
    df_brain_stats['Sex'] = None

    for _,row in df_brain_stats.iterrows():
        age = df_motion.loc[(df_motion['Session']==row['Session']) & (df_motion['DTI_ID']==1), 'Age'].values
        if len(age)==0:
            # print('Notice: cannot find age info for session {}'.format(row['Session']))
            pass
        else:
            df_brain_stats.loc[df_brain_stats['Session']==row['Session'], 'Age'] = age[0]
        
        sex = df_motion.loc[(df_motion['Session']==row['Session']) & (df_motion['DTI_ID']==1), 'Sex'].values
        if len(sex)==0:
            # print('Notice: cannot find sex info for session {}'.format(row['Session']))
            pass
        else:
            df_brain_stats.loc[df_brain_stats['Session']==row['Session'], 'Sex'] = sex[0]

    # Drop NaN rows
    df_brain_stats.dropna(inplace=True)

    return df_brain_stats


def age_prediction(atlas, DTI_scalar, value, volume, random_seed=0):

    # Load data
    df = prepare_dataframe(atlas=atlas, DTI_scalar=DTI_scalar, value=value, volume=volume)
    df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    X = df_encoded.drop(['Session','Age'], axis=1)
    y = df_encoded['Age']
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=random_seed)
    rf_mae_mean = np.mean(-cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error'))
    
    # Fully Connected Neural Network regression model (MLP)
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=5000, random_state=random_seed)
    mlp_mae_mean = np.mean(-cross_val_score(mlp_model, X, y, cv=5, scoring='neg_mean_absolute_error'))

    num_input_feat = X.shape[1] - 2
    return num_input_feat, rf_mae_mean, mlp_mae_mean