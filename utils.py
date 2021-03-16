import torch
from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1, gamma=10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        
    def forward(self, fakeB_pred, idenB, cycleB, realB):
        fakeB_target = torch.ones_like(fakeB_pred)
        advLoss = self.bce(fakeB_pred, fakeB_target)
        idenLoss = self.l1(idenB, realB)
        cycleLoss = self.l1(cycleB, realB)
        loss = self.alpha* advLoss + self.beta* idenLoss + self.gamma* cycleLoss
        return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, fakeB_pred, realB_pred):
        fakeB_target = torch.zeros_like(fakeB_pred)
        realB_target = torch.ones_like(realB_pred)
        fakeLoss = self.bce(fakeB_pred, fakeB_target)
        realLoss = self.bce(realB_pred, realB_target)
        
        loss = (fakeLoss + realLoss)/2.
        return loss