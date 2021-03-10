import os
import glob
import torch
from torch import nn
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.x = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.y = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        if len(self.x) > len(self.y):
            self.x, self.y = self.y, self.x
        self.new_perm()
        assert len(self.x) > 0, "Make sure you downloaded the horse2zebra images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.y))[:len(self.x)]

    def __getitem__(self, index):
        x = self.transform(Image.open(self.x[index % len(self.x)]))
        y = self.transform(Image.open(self.y[self.randperm[index]]))
        if x.shape[0] != 3: 
            x = x.repeat(3, 1, 1)
        if y.shape[0] != 3: 
            y = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (x - 0.5) * 2, (y - 0.5) * 2

    def __len__(self):
        return min(len(self.x), len(self.y))