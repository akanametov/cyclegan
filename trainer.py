import torch
from torch import nn

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from IPython.display import clear_output

def showImage(img_data, title):
    img_data = ((img_data + 1.)/2).detach().cpu()
    img_grid = make_grid(img_data[:4], nrow=4)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()

def ConcatParameters(parametersA, parametersB):
    return iter(list(parametersA) + list(parametersB))

class Trainer():
    def __init__(self,
                 generatorAB, generatorBA,
                 g_criterion, g_optimizer,
                 
                 discriminatorB, discriminatorA,
                 d_criterion, dB_optimizer, dA_optimizer, device='cuda:0'):
        
        self.generatorAB=generatorAB.to(device)
        self.generatorBA=generatorBA.to(device)
        self.g_criterion=g_criterion
        self.g_optimizer=g_optimizer
        
        self.discriminatorB=discriminatorB.to(device)
        self.discriminatorA=discriminatorA.to(device)
        self.d_criterion=d_criterion
        self.dB_optimizer=dB_optimizer
        self.dA_optimizer=dA_optimizer
        
    def fit(self, dataloader, epochs=10, display_step=200, device='cuda:0'):
        losses={'gloss':[], 'dloss':[], 'iter': []}
        mean_gloss = 0
        mean_dloss = 0
        t = 0
        for epoch in range(epochs):
            clear_output(wait=True)
            for realA, realB in tqdm(dataloader):

                realA = realA.to(device)
                realB = realB.to(device)
                
                # Discriminator A
                self.dA_optimizer.zero_grad()
                with torch.no_grad():
                    fakeA = self.generatorBA(realB).detach()
                fakeA_pred = self.discriminatorA(fakeA)
                realA_pred = self.discriminatorA(realA)
                dA_loss = self.d_criterion(fakeA_pred, realA_pred)
                dA_loss.backward(retain_graph=True)
                self.dA_optimizer.step()
                
                # Discriminator B
                self.dB_optimizer.zero_grad()
                with torch.no_grad():
                    fakeB = self.generatorAB(realA).detach()
                fakeB_pred = self.discriminatorB(fakeB)
                realB_pred = self.discriminatorB(realB)
                dB_loss = self.d_criterion(fakeB_pred, realB_pred)
                dB_loss.backward(retain_graph=True)
                self.dB_optimizer.step()
                
                # Generator AB and Generator BA
                self.g_optimizer.zero_grad()
                
                fakeB = self.generatorAB(realA)
                fakeB_pred = self.discriminatorB(fakeB)
                idenB = self.generatorAB(realB)
                cycleB = self.generatorAB(fakeA)
                gAB_loss = self.g_criterion(fakeB_pred, idenB, cycleB, realB)
                
                
                fakeA = self.generatorBA(realB)
                fakeA_pred = self.discriminatorA(fakeA)
                idenA = self.generatorBA(realA)
                cycleA = self.generatorBA(fakeB)
                gBA_loss = self.g_criterion(fakeA_pred, idenA, cycleA, realA)
                
                g_loss = gAB_loss + gBA_loss
                g_loss.backward()
                self.g_optimizer.step()
                
                mean_gloss += g_loss.item()/display_step
                mean_dloss += dA_loss.item()/display_step
                
                if t% display_step == 0:
                    losses['gloss'].append(mean_gloss)
                    losses['dloss'].append(mean_dloss)
                    losses['iter'].append(t)
                    print(f':::::::::::::::::  Epoch {epoch+1}  :::::::::::::::::')
                    print(f'::::::::::::::::  Iteration {t}  :::::::::::::::')
                    print(f'::::::::::: Generator loss: {mean_gloss:.5f} :::::::::::')
                    print(f'::::::::: Discriminator loss: {mean_dloss:.5f} :::::::::')
                    reals = torch.cat([realA, realB], dim=0)
                    showImage(reals, 'Input "Image A" and "Image B"')
                    fakes = torch.cat([fakeB, fakeA], dim=0)
                    showImage(fakes, 'Generated "Image B" and "Image A"')
                    mean_gloss = 0
                    mean_dloss = 0
                t += 1
        self.data=losses