import torch
from torch import nn
    
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
        adv_loss = self.bce(fake_pred, fake_target)
        iden_loss = self.l1(iden, real)
        cycle_loss = self.l1(cycle, real)
        loss = self.alpha* adv_loss + self.beta* iden_loss + self.gamma* cycle_loss
        return loss    
    
class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss