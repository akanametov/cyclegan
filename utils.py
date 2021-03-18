import torch
from torch import nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        loss = (self.bce(fake_pred, fake_target) + self.bce(real_pred, real_target))/2
        return loss
    
class GeneratorLoss(nn.Module):
    def __init__(self, alpha=1, beta=10, gamma=10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        
    def forward(self, fake_pred, iden, cycle, real):
        fake_target = torch.ones_like(fake_pred)
        advLoss = self.bce(fake_pred, fake_target)
        idenLoss = self.l1(iden, real)
        cycleLoss = self.l1(cycle, real)
        loss = self.alpha* advLoss + self.beta* idenLoss + self.gamma* cycleLoss
        return loss