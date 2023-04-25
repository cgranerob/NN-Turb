# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

#%% 1. Data exploration

### (a) import dataset
data=np.load('Train_TurbModane.npz')
S2=data['S2']
S3=data['Skew']
S4=data['Flat']
scales=data['scales']

#%% (b) format data
signals_all = np.stack((np.log(S2),S3,S4),axis=1)

#%% (c) create dataloader
import torch
from torch.utils.data import DataLoader, Dataset

class TrajDataSet(Dataset):
    def __init__(self,  traj, transform=None):
        self.traj = traj
        self.transform = transform

    def __len__(self):
        return self.traj.shape[0]

    def __getitem__(self, idx):
        # select coordinates
        sample = self.traj[idx,:,:]
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        if(cuda):
            return torch.FloatTensor(sample).cuda()
        else:
            return torch.FloatTensor(sample)

# hyperparameters
batch_size = 16
batches=signals_all.shape[0]/batch_size

## reduce size dataset
train_set = TrajDataSet(signals_all, transform= ToTensor())
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers = 0, shuffle = True, drop_last=False)

#%% 2.1 GAN Training 

from torch import nn
from torch import optim
import progressbar

from analyseIncrsTorchcuda import analyseIncrsTorchcuda_vp

cuda = True if torch.cuda.is_available() else False
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool1d(2, ceil_mode=False)
        self.avgpoolc = nn.AvgPool1d(2, ceil_mode=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')

        self.cnn1 = nn.Sequential( 
            nn.Conv1d(1, 16, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            )
        self.cnn8 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size = 8, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            )
        self.cnn16 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 16, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnn32 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 32, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnn64 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnntrans64 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnntrans32 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 32, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.cnntrans16 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size = 16, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            )
        self.cnntrans8 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size = 8, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            )
        self.cnntrans4 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            )
        self.cnntrans2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size = 2, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            )
        self.cnntrans1 = nn.Sequential(
            nn.ConvTranspose1d(16, 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            )
        self.bridge1 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.bridge2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 128, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.bridge3 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 128, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        self.bridge4 = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size = 64, stride = 1, padding = 0, bias = False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            )
        
    def forward(self, z):    
        residual1  = self.cnn1(z)
        out = residual1 #= out->  size
        
        residual2  = self.cnn2(out)
        out = residual2 #= out -> size/2
        
        out = self.avgpoolc(out)
        residual4  = self.cnn4(out)
        out = residual4 #= out ->size/4
        
        out = self.avgpoolc(out)
        residual8  = self.cnn8(out)
        out = residual8 #= out -> size=8
        
        out = self.avgpoolc(out)
        residual16  = self.cnn16(out)
        out = residual16 #= out -> size/16
        
        out = self.avgpoolc(out)
        residual32  = self.cnn32(out)
        out = residual32 #= out size/32
        
        out = self.avgpoolc(out)
        out  = self.cnn64(out)
        
        #Bridge
        out = self.bridge1(out)
        out = self.bridge4(out)
        #End of Bridge
        
        out  = self.cnntrans64(out)
        out = self.upsample(out)
        out = out + residual32 # -> size/32
        
        out  = self.cnntrans32(out)
        out  =self.upsample(out)
        out = out[:,:,0:-1] + residual16 # -> size/16
        
        out  = self.cnntrans16(out)
        out  =self.upsample(out)
        out  = out + residual8 #-> size/8

        out  = self.cnntrans8(out)
        out  =self.upsample(out)
        out  = out[:,:,0:-1] + residual4 #-> size/4

        out  = self.cnntrans4(out)
        out  =self.upsample(out)
        out  = out[:,:,0:-1] + residual2 #-> size/2

        out  = self.cnntrans2(out)
        out  = out + residual1 #->size

        out  = self.cnntrans1(out)
        return out

#%%

def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. This follows the specification from the DCGAN paper.
    https://arxiv.org/pdf/1511.06434.pdf
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# create network
generator = CNNGenerator().to(dev)

# weight initialization
generator = generator.apply(weights_init)

# define loss and optimizers
criterion = nn.MSELoss().to(dev)
criterion2 = nn.KLDivLoss().to(dev)
lr = 0.0002
optim_g = optim.Adam(generator.parameters(),lr= lr, betas=(0.5, 0.999))

print("Model  ", next(generator.parameters()).is_cuda)

#%% 2.2 GAN Training 

#Hyperparameters
alpha=1
beta=1
gamma=1
lamda=0.1
# Size of time series
Nts=2**15

# Train the model
nb_epoch = 2000
# Initialize
cout=torch.zeros((nb_epoch), device=dev)
cout1=torch.zeros((nb_epoch), device=dev)
cout2=torch.zeros((nb_epoch), device=dev)
cout3=torch.zeros((nb_epoch), device=dev)
cout4=torch.zeros((nb_epoch), device=dev)

epoch = 0
for i in progressbar.progressbar(range(nb_epoch)):
    epoch +=1
    if epoch ==100:
        for g in optim_g.param_groups:
            g['lr'] = 0.0001
    if epoch ==1000:
        for g in optim_g.param_groups:
            g['lr'] = 0.00005

    for batch, x in enumerate(train_loader):
        
        #### TRAIN GENERATOR
        generator.zero_grad()
        z = torch.randn((batch_size, 1, Nts), device=dev)

        generated = generator(z)
        
        # Estimation of structure functions    
        sgenerated=analyseIncrsTorchcuda_vp(torch.cumsum(generated,dim=2),scales, dev).to(dev)

        loss1 = criterion(sgenerated[:,0,:], x[:,0,:])
        loss2 = criterion(sgenerated[:,1,:], x[:,1,:])
        loss3 = criterion(sgenerated[:,2,:], x[:,2,:])
        loss4 = criterion((torch.cumsum(generated,dim=2)-torch.mean(torch.cumsum(generated,dim=2), dim=2, keepdim=True))/torch.std(torch.cumsum(generated,dim=2),dim=2, keepdim=True),z)
        loss= alpha*loss1 + beta*loss2 +gamma*loss3 + lamda*loss4 
        loss.backward()
        optim_g.step()
        '''
        '''
        ll=+loss
        ll1=+loss1
        ll2=+loss2
        ll3=+loss3
        ll4=+loss4
    cout[i]=ll/batches
    cout1[i]=ll1/batches
    cout2[i]=ll2/batches
    cout3[i]=ll3/batches
    cout4[i]=ll4/batches
    if epoch%100 == 0:
        print('\nEpoch [{}/{}] -----------------------------------------------------------------------------'
            .format(epoch+1, nb_epoch))

        PATH='NNTurb_epoch_'+str(epoch)+'.pt'
        torch.save({
            #'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            #'optimizer_state_dict': optim_g.state_dict(),
            #'loss': loss,
            }, PATH)

cout = cout.cpu().detach().numpy()
cout1 = cout1.cpu().detach().numpy()
cout2 = cout2.cpu().detach().numpy()
cout3 = cout3.cpu().detach().numpy()
cout4 = cout4.cpu().detach().numpy()

np.savez('Loss_NNTurb.npz',cout=cout,cout1=cout1,cout2=cout2,cout3=cout3,cout4=cout4)
