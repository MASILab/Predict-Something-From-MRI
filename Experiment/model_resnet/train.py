import argparse
import torch
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dataset import AgePredictionDataset
from models import *
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def select_model(model_name):
    if model_name == 'resnet10_noMLP':
        print('Loading {}'.format(model_name))
        return resnet10_noMLP()
    elif model_name == 'resnet10_MLP_64':
        print('Loading {}'.format(model_name))
        return resnet10_MLP_64()
    else:
        print('Model not found. Please see models.py for the full list of models')


if __name__ == "__main__":
    # user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='resnet10_noMLP', help="Name of the model, see models.py for the full list of models")
    parser.add_argument("--fold", nargs='+', type=int, default=[1,2,3,4,5], help="Fold(s) to train, e.g.'--fold 1 2 3 4 5', default to 1 2 3 4 5")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs, default to 100")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size, default to 8")
    parser.add_argument("--model_save_dir", type=str, default='/home-local/Projects/Predict_Age_Models', help="Directory to save the model weights, default to /home-local/Projects/Predict_Age_Models")
    args = parser.parse_args()

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training/validation data and 5-fold cross validation
    datasplit_folder = Path('/nfs/masi/gaoc11/projects/Predict-Something-From-MRI/Experiment/model_resnet/data')
    train_csv = datasplit_folder / 'train.csv'
    df = pd.read_csv(train_csv)

    for fold_idx in args.fold:

        # subject-level splitting
        subjects_train = np.load((datasplit_folder/'train_subjects_t_fold_{}.npy'.format(fold_idx)), allow_pickle=True)
        subjects_val = np.load((datasplit_folder/'train_subjects_v_fold_{}.npy'.format(fold_idx)), allow_pickle=True)
        
        df_train = df[df['Subject'].isin(subjects_train)]
        df_val = df[df['Subject'].isin(subjects_val)]
        
        dataset_train = AgePredictionDataset(df_train)
        dataset_val = AgePredictionDataset(df_val)
        
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size*2, shuffle=False, num_workers=4, pin_memory=True)

        print("Start fold-{}\nTraining set:\n{}\nValidation set:\n{}\n".format(fold_idx, df_train, df_val))
        
        # TODO: think more on this block, there are many options
        model = select_model(args.model_name).to(device, non_blocking=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[2, 6, 14, 30], gamma=0.2, last_epoch=-1, verbose=True)
        loss_fn = torch.nn.MSELoss()

        best_val_loss = float('inf')
        model_save_dir = Path(args.model_save_dir) / args.model_name / "fold-{}".format(fold_idx)
        if not model_save_dir.is_dir():
            subprocess.run(['mkdir', '-p', model_save_dir])
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for fa, md, sex, age in tqdm(dataloader_train):                
                fa, md, sex, age = fa.to(device, non_blocking=True), md.to(device, non_blocking=True), sex.to(device, non_blocking=True), age.to(device, non_blocking=True)
                input_img = torch.cat((fa,md), dim=1)
                
                optimizer.zero_grad()

                with torch.autocast():
                    output = model(input_img, sex)
                    loss = loss_fn(output, age.view(-1, 1))
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            
            epoch_loss = epoch_loss / len(dataloader_train)
            writer.add_scalar('Train/Loss', epoch_loss, epoch)
                
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for fa_val, md_val, sex_val, age_val in dataloader_val:
                    fa_val, md_val, sex_val, age_val = fa_val.to(device, non_blocking=True), md_val.to(device, non_blocking=True), sex_val.to(device, non_blocking=True), age_val.to(device, non_blocking=True)

                    input_img_val = torch.cat((fa_val,md_val), dim=1)
                    output_val = model(input_img_val, sex_val)
                    loss_val = loss_fn(output_val, age_val.view(-1, 1))
                    
                    val_loss += loss_val.item()

            val_loss = val_loss / len(dataloader_val)
            writer.add_scalar('Val/Loss', val_loss, epoch)

            print("Epoch: {}\tTrain Loss: {}\tValidation Loss: {}".format(epoch, epoch_loss, val_loss))
            
            # Check if this model is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = model_save_dir / "model_fold-{}_epoch-{}.pth".format(fold_idx, epoch)
                torch.save(model.state_dict(), save_path)
                print(f'Saved improved model to {save_path} at epoch {epoch} with validation loss {best_val_loss}')

writer.close()