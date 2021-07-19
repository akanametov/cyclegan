import glob
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from .utils import download_and_extract

class Apple2Orange(Dataset):
    url="https://github.com/akanametov/cyclegan/releases/download/1.0/apple2orange.zip"
    def __init__(self,
                 root: str='.',
                 transform=None,
                 download: bool=True,
                 mode: str='train',
                 direction: str='A2B'):
        if download:
            _ = download_and_extract(root, self.url)
        self.root=root
        self.filesA=sorted(glob.glob(f"{root}/apple2orange/{mode}A/*.jpg"))
        self.filesB=sorted(glob.glob(f"{root}/apple2orange/{mode}B/*.jpg"))
        random.shuffle(self.filesB)
        self.transform=transform
        self.download=download
        self.mode=mode
        self.direction=direction
        
    def __len__(self,):
        return min(len(self.filesA), len(self.filesB))
    
    def __getitem__(self, idx):
        imgA = Image.open(self.filesA[idx]).convert('RGB')
        imgB = Image.open(self.filesB[idx]).convert('RGB')
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        if self.direction == 'A2B':
            return imgA, imgB
        else:
            return imgB, imgA