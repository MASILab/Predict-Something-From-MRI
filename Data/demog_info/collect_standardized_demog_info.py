# Collect demographic/diagnosis information for the remaining data after quality check
# 
# Author: Chenyu Gao
# Date: Jul 13, 2023

import pandas as pd

# Spreadsheet of remaining data after quality check
df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/quality_check/quality_check_results.csv')

# Information to collect
df['Age'] = None
df['Sex'] = None
df['Diagnosis'] = None

#
dataset = 'BIOCARD'
print("Collecting information for {} data\n".format(dataset))
df_demog = pd.read_excel('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/raw/BIOCARD_External_Data_2022.08.08/BIOCARD_Demographics_Limited_Data_2022.05.10.xlsx')
df_diagnosis = pd.read_excel('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/raw/BIOCARD_External_Data_2022.08.08/BIOCARD_DiagnosisData_Limited_2022.05.14.xlsx')

# Look up table for standardized values
sex2standard = {1:'male', 2:'female'}

diagnosis2standard = {'NORMAL': 'normal',
                      'MCI': 'MCI',  
                      'IMPAIRED NOT MCI': 'impaired (not MCI)',
                      'DEMENTIA': 'dementia'}

for _,row in df.loc[df['Dataset']==dataset].iterrows():
    
    # Labels for referencing the demographic and diagnosis spreadsheet
    JHUANONID = row['Subject'].replace('sub-','')
    current_year = 2000 + int(row['Session'].split('_ses-')[1][0:2])
    if current_year >= 2023:
        print("Warning: data from the future! The calculation is wrong!")
        break
    
    # Retrieve information if exists
    try:
        age = current_year - df_demog.loc[df_demog['JHUANONID']==JHUANONID,'BIRTHYEAR'].values[0]
    except:
        age = None
        print("Notice: missing age info for {}".format(JHUANONID))
    
    try:
        sex = df_demog.loc[df_demog['JHUANONID']==JHUANONID, 'SEX'].values[0]
        sex = sex2standard[sex]
    except:
        sex = None
        print("Notice: missing sex info for {}".format(JHUANONID))
    
    try:
        diagnosis = df_diagnosis.loc[(df_diagnosis['JHUANONID']==JHUANONID)&(df_diagnosis['STARTYEAR']==current_year), 'DIAGNOSIS'].values[-1]  # if there are multiple matches, choose the worst one
        diagnosis = diagnosis2standard[diagnosis]
    except:
        diagnosis = None
        print("Notice: missing diagnosis info for {}".format(JHUANONID))
        
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Age'] = age
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Sex'] = sex
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Diagnosis'] = diagnosis


#
dataset = 'ICBM'
print("Collecting information for {} data\n".format(dataset))
df_demog = pd.read_excel('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/raw/ICBM_Clinical_Data_18APR2012.xlsx',
                         header=0, usecols="A:F",skiprows=[1,2,192,193,194,195,196,197,198,199,200,201])

sex2standard = {'Female': 'female',
                'Male': 'male'}

for _,row in df.loc[df['Dataset']==dataset].iterrows():
    subject_id = 'UTHC_' + row['Subject'].replace('sub-','')[-4:]
    
    try:
        age = df_demog.loc[df_demog['Subject ID']==subject_id, 'Age'].values[0]
    except:
        age = None
        print("Notice: missing age info for {}".format(subject_id))
    
    try:
        sex = df_demog.loc[df_demog['Subject ID']==subject_id, 'Gender'].values[0]
        sex = sex2standard[sex]
    except:
        sex = None
        print("Notice: missing sex info for {}".format(subject_id))
    
    diagnosis = 'normal'

    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Age'] = age
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Sex'] = sex
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Diagnosis'] = diagnosis
    
#        
dataset = 'BLSA'
print("Collecting information for {} data\n".format(dataset))
df_demog = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/raw/BLSA_demog.csv')

sex2standard = {1: 'male', 0: 'female'}
diagnosis2standard = {0: 'normal',
                      0.5: 'MCI',
                      -0.5: 'impaired (not MCI)',
                      1: 'dementia'}

for _,row in df.loc[df['Dataset']==dataset].iterrows():
    label = "BLSA_{}_{}-{}_{}".format(row['Subject'].split('BLSA')[1],
                                      row['Session'].split('_ses-')[1][0:2],
                                      row['Session'].split('_ses-')[1][2],
                                      row['Session'].split('scanner')[1])
    try:
        age = df_demog.loc[df_demog['labels']==label, 'Age'].values[0]
    except:
        age = None
        print("Notice: missing age info for {}".format(row['Session']))
    
    try:
        sex = df_demog.loc[df_demog['labels']==label, 'sex'].values[0]
        sex = sex2standard[sex]
    except:
        sex = None
        print("Notice: missing sex info for {}".format(row['Session']))
    
    try:
        diagnosis = df_demog.loc[df_demog['labels']==label, 'dxatvi'].values[0]
        diagnosis = diagnosis2standard[diagnosis]
    except:
        diagnosis = None
        print("Notice: missing diagnosis info for {}".format(row['Session']))
        
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Age'] = age
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Sex'] = sex
    df.loc[(df['Dataset']==dataset)&(df['Session']==row['Session']), 'Diagnosis'] = diagnosis       

# Save to csv
# df.dropna(axis=0, inplace=True)
df.to_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/demog_info/data_all_with_subject_info.csv', index=False)
