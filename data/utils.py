import os
import glob
import torch
import random
from torch import nn
from PIL import Image

class DataSet(torch.utils.data.Dataset):
    def __init__(self, pathA, pathB, transform=None):
        self.pathA = pathA
        self.pathB = pathB
        self.transform = transform
        
    def __len__(self,):
        return min(len(self.pathA), len(self.pathB))
    
    def __getitem__(self, idx):
        imgA = Image.open(self.pathA[idx])
        imgB = Image.open(self.pathB[idx])
        if idx == (len(self)-1):
            random.shuffle(self.pathA)
            random.shuffle(self.pathB)
        return self.transform(imgA), self.transform(imgB)