import argparse
import torch
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from models_copy import *
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