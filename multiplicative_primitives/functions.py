import torch
import torch.nn as nn
import torch.nn.functional as F

class Trunk(nn.Module):
    def __init__(self, dims):
        super(Trunk, self).__init__()
        self.dims = dims
        self.dims = dims
        self.layers = nn.ModuleList()
        for i in range(len(self.dims[:-1])):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x