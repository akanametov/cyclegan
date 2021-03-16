import torch
from torch import nn

class ConvIn(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3,
                 stride=2, padding=1, padding_mode='reflect'):
        super().__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, padding_mode=padding_mode)
        self.insnorm=nn.InstanceNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.insnorm(x)
        return x
    
class ConvReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3,
                 stride=2, padding=1, padding_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, padding_mode=padding_mode)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
    
class ConvLReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3,
                 stride=2, padding=1, padding_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, padding_mode=padding_mode)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.lrelu(x)
        return x
    
    
class ConvInReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3,
                 stride=2, padding=1, padding_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, padding_mode=padding_mode)
        self.insnorm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.insnorm(x)
        x = self.relu(x)
        return x
    
    
class ConvInLReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel_size=3,
                 stride=2, padding=1, padding_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, padding_mode=padding_mode)
        self.insnorm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.insnorm(x)
        x = self.lrelu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_in_relu = ConvInReLU(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_in = ConvIn(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        fx = self.conv_in_relu(x)
        fx = self.conv_in(fx)
        return fx + x
    
    
class UpConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2,
                 padding=1, output_padding=1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.upconv(x)
        x = self.relu(x)
        return x
    
class UpConvInReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2,
                 padding=1, output_padding=1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, output_padding=output_padding)
        self.insnorm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.upconv(x)
        x = self.insnorm(x)
        x = self.relu(x)
        return x
    
class ConvTanh(nn.Module):
    def __init__(self, in_channels, out_channels,
                       kernel_size=7, stride=1,
                       padding=3, padding_mode='reflect'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, padding_mode=padding_mode)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        return x