import torch
from torch import nn

def initialize_weights(layer, mean=0., std=0.02):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, mean, std)