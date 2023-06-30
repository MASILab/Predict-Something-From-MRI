import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import pdb

# Load the dataset
df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Data/toy_regression_spreadsheet/toy_dataset.csv')

# Preprocess the data
df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)  # actually it's already one-hot encoding

X = df_encoded.drop(['Session','Age'], axis=1)
y = df_encoded['Age']

# Define the different training data sizes to test
training_data_sizes = [1, 2, 3, 4, 5, 10, 50, 100, 200, 500, 1000, 1500, 2000, 2500]
# training_data_sizes = [1, 5]

# Initialize lists to store the accuracy values
svm_mae_all = []
rf_mae_all = []
mlp_mae_all = []

# Iterate over the different training data sizes
for size in training_data_sizes:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
    
    # Support Vector Machines regression model (SVM)
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_mae = mean_absolute_error(y_test, svm_predictions)
    svm_mae_all.append(svm_mae)
    
    # Random Forest regression model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_mae_all.append(rf_mae)
    
    # Fully Connected Neural Network regression model (MLP)
    mlp_model = MLPRegressor(hidden_layer_sizes=(100,10), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    mlp_mae = mean_absolute_error(y_test, mlp_predictions)
    mlp_mae_all.append(mlp_mae)

# Print the accuracies for each training data size
for i, size in enumerate(training_data_sizes):
    print(f"Training Data Size: {size}")
    print(f"SVM MAE: {svm_mae_all[i]}")
    print(f"Random Forest MAE: {rf_mae_all[i]}")
    print(f"MLP MAE: {mlp_mae_all[i]}")
    print()
    
# Visualize the MAE for all methods
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
ax.plot(training_data_sizes, svm_mae_all, label='SVM')
ax.plot(training_data_sizes, rf_mae_all, label='Random Forest')
ax.plot(training_data_sizes, mlp_mae_all, label='Shallow FCN (MLP)')
ax.set_xlabel('Number of Training Data')
ax.set_ylabel('Mean Absolute Error in Testing (years)')
ax.set_title('Predict age from volume, FA/MD/RD/AD mean of ROIs by Eve-1, and sex')
ax.legend()

fig.savefig('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/toy_regression/toy_regression_experiment.png')