import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import numpy as np
import pdb
import pickle

# Load the dataset
df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/toy_regression_spreadsheet/toy_dataset.csv')

# Preprocess the data
df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)  # actually it's already one-hot encoding

X = df_encoded.drop(['Session','Age'], axis=1)
y = df_encoded['Age']

# Define the different training data sizes to test
training_data_sizes = [1, 2, 3, 4, 5, 10, 50, 100, 200, 500, 1000, 1500, 2000, 2500]
# training_data_sizes = [50, 100]

# Initialize lists to store the accuracy values
svm_mae_train = []
rf_mae_train = []
mlp_mae_train = []

svm_mae_test = []
rf_mae_test = []
mlp_mae_test = []

# Iterate over the different training data sizes
for size in tqdm(training_data_sizes):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=0)
    
    # Support Vector Machines regression model (SVM)
    svm_model = SVR()
    if size >=5:
        svm_scores = -cross_val_score(svm_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        svm_mae_train.append(svm_scores)
    
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_mae = mean_absolute_error(y_test, svm_predictions)
    svm_mae_test.append(svm_mae)
    
    # Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    if size >=5:
        rf_scores = -cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        rf_mae_train.append(rf_scores)

    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mae_test.append(rf_mae)
    
    # Fully Connected Neural Network regression model (MLP)
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,10), max_iter=5000, random_state=0)
    if size >=5:
        mlp_scores = -cross_val_score(mlp_model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mlp_mae_train.append(mlp_scores)

    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    mlp_mae = mean_absolute_error(y_test, mlp_predictions)
    mlp_mae_test.append(mlp_mae)

# Save the variables in case the script crashes
with open('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/toy_regression/tmp_variable.pickle', 'wb') as f:
    pickle.dump((svm_mae_train, svm_mae_test, rf_mae_train, rf_mae_test, mlp_mae_train, mlp_mae_test), f)
# with open('variables.pickle', 'rb') as f:
#     svm_mae_train, svm_mae_test, rf_mae_train, rf_mae_test, mlp_mae_train, mlp_mae_test = pickle.load(f)

# Visualize the MAE for all methods
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

# SVM
svm_mae_train = np.array(svm_mae_train)
ax.plot(training_data_sizes[-svm_mae_train.shape[0]:], np.mean(svm_mae_train, axis=1), marker='o', color='tab:blue', label='SVM (5-fold cross val)')
ax.fill_between(training_data_sizes[-svm_mae_train.shape[0]:],
                np.subtract(np.mean(svm_mae_train, axis=1), np.std(svm_mae_train, axis=1)), 
                np.add(np.mean(svm_mae_train, axis=1), np.std(svm_mae_train, axis=1)),
                color='tab:blue',
                alpha=0.2)
ax.plot(training_data_sizes, svm_mae_test, linestyle='--', marker='o', color='tab:blue', label='SVM (testing)')

# Random Forest
rf_mae_train = np.array(rf_mae_train)
ax.plot(training_data_sizes[-rf_mae_train.shape[0]:], np.mean(rf_mae_train, axis=1), marker='o', color='tab:orange', label='Random Forest (5-fold cross val)')
ax.fill_between(training_data_sizes[-rf_mae_train.shape[0]:],
                np.subtract(np.mean(rf_mae_train, axis=1), np.std(rf_mae_train, axis=1)), 
                np.add(np.mean(rf_mae_train, axis=1), np.std(rf_mae_train, axis=1)),
                color='tab:orange',
                alpha=0.2)
ax.plot(training_data_sizes, rf_mae_test, linestyle='--', marker='o', color='tab:orange', label='Random Forest (testing)')

# Shallow MLP
mlp_mae_train = np.array(mlp_mae_train)
ax.plot(training_data_sizes[-mlp_mae_train.shape[0]:], np.mean(mlp_mae_train, axis=1), marker='o', color='tab:green', label='Shallow MLP (5-fold cross val)')
ax.fill_between(training_data_sizes[-mlp_mae_train.shape[0]:],
                np.subtract(np.mean(mlp_mae_train, axis=1), np.std(mlp_mae_train, axis=1)), 
                np.add(np.mean(mlp_mae_train, axis=1), np.std(mlp_mae_train, axis=1)),
                color='tab:green',
                alpha=0.2)
ax.plot(training_data_sizes, mlp_mae_test, linestyle='--', marker='o', color='tab:green', label='Shallow MLP (testing)')

ax.set_xlabel('Number of Training Data')
ax.set_ylabel('Mean Absolute Error (years)')
ax.set_title('Predict age from volume, FA/MD/RD/AD mean of ROIs by Eve-1, and sex')
ax.legend()

fig.savefig('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/toy_regression/toy_regression_experiment_cross_val_seed_0.png')