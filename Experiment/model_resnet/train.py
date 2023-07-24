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

def select_model(model_name):
    if model_name == 'resnet10_noMLP':
        print('Loading {}'.format(model_name))
        return resnet10_noMLP()
    elif model_name == 'resnet10_MLP_64':
        print('Loading {}'.format(model_name))
        return resnet10_MLP_64()
    elif model_name == 'resnet18_noMLP':
        print('Loading {}'.format(model_name))
        return resnet18_noMLP()
    elif model_name == 'resnet18_MLP_64':
        print('Loading {}'.format(model_name))  
        return resnet18_MLP_64()
    elif model_name == 'resnet34_noMLP':
        print('Loading {}'.format(model_name))  
        return resnet34_noMLP()
    elif model_name == 'resnet34_MLP_64':
        print('Loading {}'.format(model_name))
        return resnet34_MLP_64()
    else:
        print('Model not found. Please see models.py for the full list of models')

if __name__ == "__main__":
    # user inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='resnet10_noMLP', help="Name of the model, see models.py for the full list of models")
    parser.add_argument("--fold", nargs='+', type=int, default=[1,2,3,4,5], help="Fold(s) to train, e.g.'--fold 1 2 3 4 5', default to 1 2 3 4 5")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs, default to 30")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the dataloader, default to 2")
    parser.add_argument("--bsize_factor", type=int, default=2, help="Update weights for every other bsize_factor batch to mimic larger batchsize, default to 2")
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
        
        # tensorboard
        writer = SummaryWriter(log_dir="runs/{}_fold-{}".format(args.model_name, fold_idx))

        # subject-level splitting
        subjects_train = np.load((datasplit_folder/'train_subjects_t_fold_{}.npy'.format(fold_idx)), allow_pickle=True)
        subjects_val = np.load((datasplit_folder/'train_subjects_v_fold_{}.npy'.format(fold_idx)), allow_pickle=True)
        
        df_train = df[df['Subject'].isin(subjects_train)]
        df_val = df[df['Subject'].isin(subjects_val)]
        
        dataset_train = AgePredictionDataset(df_train)
        dataset_val = AgePredictionDataset(df_val)
        
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("Start fold-{}\nTraining set:\n{}\nValidation set:\n{}\n".format(fold_idx, df_train, df_val))
        
        # TODO: think more on this block, there are many options
        model = select_model(args.model_name).to(device, non_blocking=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=1e-3, 
                                            epochs=num_epochs,
                                            steps_per_epoch=round(len(dataset_train)/(batch_size*args.bsize_factor)), 
                                            cycle_momentum=True)
        loss_fn = torch.nn.MSELoss()

        best_val_loss = float('inf')
        model_save_dir = Path(args.model_save_dir) / args.model_name / "fold-{}".format(fold_idx)
        if not model_save_dir.is_dir():
            subprocess.run(['mkdir', '-p', model_save_dir])
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            for i, (fa, md, sex, age) in enumerate(tqdm(dataloader_train)):
                fa, md, sex, age = fa.to(device, non_blocking=True), md.to(device, non_blocking=True), sex.to(device, non_blocking=True), age.to(device, non_blocking=True)
                input_img = torch.cat((fa,md), dim=1)
                
                with torch.autocast('cuda'):
                    output = model(input_img, sex)
                    loss = loss_fn(output, age.view(-1, 1))
                    epoch_loss += loss.item()

                loss /= args.bsize_factor
                loss.backward()

                if (i+1) % args.bsize_factor == 0 or (i+1) == len(dataloader_train):
                    optimizer.step()
                    optimizer.zero_grad()
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

            print("Epoch: {}\tCurrent LR: {}\tTrain Loss: {}\tValidation Loss: {}".format(epoch, scheduler.get_last_lr(), epoch_loss, val_loss))
            
            # Check if this model is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = model_save_dir / f"model_fold-{fold_idx}_epoch-{epoch}_valloss-{best_val_loss:.4f}.pth"
                torch.save(model.state_dict(), save_path)
                print(f'Saved improved model to {save_path} at epoch {epoch} with validation loss {best_val_loss}')

        writer.close()