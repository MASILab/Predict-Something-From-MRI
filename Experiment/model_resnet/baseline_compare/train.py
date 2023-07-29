import pdb
from model import MLP
from dataset import ROIBasedAgePredictionDataset
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
random_seed = 42

# hyperparams
batch_size = 64
num_epochs = 500
lr = 1e-3
list_hidden_layer_sizes = [
    [64, 32, 8], [128, 64, 8],
    [32, 16, 8], [32, 8], [64, 32], [128, 64],
    [128, 32, 8],
    [32], [64], [128],
]
model_save_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/weights')

# train/val set (all)
df = pd.read_csv('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_baseline_ROI-based-measure/data/ROIbased_measure_train.csv')
df = df.fillna(df.median(numeric_only=True))  # previously I used dropna, but that will skip "challenging" samples (unfair for ResNet)
df.reset_index(drop=True, inplace=True)
df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# 5-fold cross validation
path_five_fold_subject_split_root = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data/')

for hidden_layer_sizes in tqdm(list_hidden_layer_sizes):
    model_name = "MLP-" + '-'.join(map(str, hidden_layer_sizes))
    
    for fold_id in [1,2,3,4,5]:
        # tensorboard
        writer = SummaryWriter(log_dir="/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/baseline_compare/runs/{}_fold-{}".format(model_name, fold_id))
        
        # subject-level splitting
        subjects_train = np.load((path_five_fold_subject_split_root/f'train_subjects_t_fold_{fold_id}.npy'), allow_pickle=True)
        subjects_val = np.load((path_five_fold_subject_split_root/f'train_subjects_v_fold_{fold_id}.npy'), allow_pickle=True)
        df_train = df_encoded[df['Subject'].isin(subjects_train)]
        df_val = df_encoded[df['Subject'].isin(subjects_val)]
        
        # dataloader
        dataset_train = ROIBasedAgePredictionDataset(df_train)
        dataset_val = ROIBasedAgePredictionDataset(df_val)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        
        # model
        torch.manual_seed(random_seed)
        model = MLP(hidden_layer_sizes=hidden_layer_sizes)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=num_epochs, steps_per_epoch=len(dataloader_train), cycle_momentum=True)

        # Train the model
        best_val_loss = float('inf')
        model_save_dir = model_save_root / model_name / f"fold-{fold_id}"
        if not model_save_dir.is_dir():
            subprocess.run(['mkdir', '-p', model_save_dir])
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            
            for X_train, y_train in dataloader_train:
                # pdb.set_trace()
                output = model(X_train)
                loss = criterion(output, y_train.view(-1, 1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(dataloader_train)
            writer.add_scalar('Train/Loss', epoch_loss, epoch)
                
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in dataloader_val:

                    output_val = model(X_val)
                    loss_val = criterion(output_val, y_val.view(-1, 1))
                    
                    val_loss += loss_val.item()

            val_loss = val_loss / len(dataloader_val)
            writer.add_scalar('Val/Loss', val_loss, epoch)

            print(f"{model_name} Epoch: {epoch}\tCurrent LR: {scheduler.get_last_lr()}\tTrain Loss: {epoch_loss}\tValidation Loss: {val_loss}")
            
            # Check if this model is the best so far
            if (val_loss < best_val_loss) and (epoch > round(num_epochs/2)):
                best_val_loss = val_loss
                save_path = model_save_dir / f"model_fold-{fold_id}_epoch-{epoch}_valloss-{best_val_loss:.4f}.pth"
                torch.save(model.state_dict(), save_path)
                print(f'Saved improved model to {save_path} at epoch {epoch} with validation loss {best_val_loss}')

        writer.close()
