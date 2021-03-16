import torch
from torch import nn

def initialize_weights(layer, mean=0.0, std=0.02):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, mean, std)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, mean, std)
        torch.nn.init.constant_(layer.bias, 0)