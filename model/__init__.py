import torch
from torch import nn

from .modules import ConvInReLU, ResidualBlock, UpConvInReLU, ConvTanh
from .modules import ConvLReLU, ConvInLReLU


class Generator(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=3):
        super().__init__()
        
        self.Input=nn.Conv2d(in_channels, hid_channels, kernel_size=7,
                             stride=1, padding=3, padding_mode='reflect')
        
        self.DownStage=nn.Sequential()
        self.DownStage.add_module('block0', ConvInReLU(hid_channels, 2*hid_channels))
        self.DownStage.add_module('block1', ConvInReLU(2*hid_channels, 4*hid_channels))
        
        self.ResStage=nn.Sequential()
        self.ResStage.add_module('block0', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block1', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block2', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block3', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block4', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block5', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block6', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block7', ResidualBlock(4*hid_channels))
        self.ResStage.add_module('block8', ResidualBlock(4*hid_channels))
        
        self.UpStage = nn.Sequential()
        self.UpStage.add_module('block0', UpConvInReLU(4*hid_channels, 2*hid_channels))
        self.UpStage.add_module('block1', UpConvInReLU(2*hid_channels, hid_channels))
        
        self.Output=ConvTanh(hid_channels, out_channels)
        
    def forward(self, x):
        x = self.Input(x)
        x = self.DownStage(x)
        x = self.ResStage(x)
        x = self.UpStage(x)
        x = self.Output(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, out_channels=1):
        super().__init__()
        self.Input=nn.Conv2d(in_channels, hid_channels, kernel_size=7,
                             stride=1, padding=3, padding_mode='reflect')
        
        self.Base=nn.Sequential()
        self.Base.add_module('block0', ConvLReLU(    hid_channels, 2*hid_channels, kernel_size=4))
        self.Base.add_module('block1', ConvInLReLU(2*hid_channels, 4*hid_channels, kernel_size=4))
        self.Base.add_module('block2', ConvInLReLU(4*hid_channels, 8*hid_channels, kernel_size=4))
        
        self.Output=nn.Conv2d(8*hid_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.Input(x)
        x = self.Base(x)
        x = self.Output(x)
        return x
