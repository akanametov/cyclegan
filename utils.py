import torch
from torch import nn

class AdverserialLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss()):
        super().__init__()
        self.loss_fn = criterion

    def forward(self, pred, target):
        return self.loss_fn(pred, target)

class IdentityLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss()):
        super().__init__()
        self.loss_fn = criterion
        
    def forward(self, pred, target):
        return self.loss_fn(pred, target)
    
class CycleLoss(nn.Module):
    def __init__(self, criterion=nn.L1Loss()):
        super().__init__()
        self.loss_fn = criterion
        
    def forward(self, pred, target):
        return self.loss_fn(pred, target)
    


class DiscriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.adv_loss = AdverserialLoss()
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        loss = (self.adv_loss(fake_pred, fake_target)\
              + self.adv_loss(real_pred, real_target))/2.
        return loss
    

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1, gamma=10):
        super().__init__()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.adv_loss = AdverserialLoss()
        self.iden_loss = IdentityLoss()
        self.cycle_loss = CycleLoss()
    
    def forward(self, realX, fakeY_pred, idenX, fakeX):
        fakeY_target = torch.ones_like(fakeY_pred)
        adv_loss = self.adv_loss(fakeY_pred, fakeY_target)
        iden_loss = self.iden_loss(idenX, realX)
        cycle_loss = self.cycle_loss(fakeX, realX)
        
        loss = (self.alpha* adv_loss\
              + self.beta* iden_loss\
              + self.gamma* cycle_loss)
        return loss