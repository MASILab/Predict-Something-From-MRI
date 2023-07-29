import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_size = 537, output_size = 1, hidden_layer_sizes = [128, 64]):
        super(MLP, self).__init__()
        layers = []
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size

        layers.append(nn.Linear(input_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)