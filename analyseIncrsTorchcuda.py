#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 23:14:12 2022

@author: administrateur
"""

import torch

def analyseIncrsTorchcuda(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
    '''      
    Struc = torch.zeros((signal.shape[0],3,len(scales)), dtype=torch.float32, device=device)

    # We normalize the image by centering and standarizing it
    Nreal=signal.size()[0]
    tmp = torch.zeros(signal.shape, device=device)    
    for ir in range(Nreal):
        nanstdtmp = torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp[ir,0,:] = (signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

    for idx, scale in enumerate(scales):
        incrs=tmp[:,:,scale:]-tmp[:,:,:-scale]
        Struc[:,0,idx] = torch.log(torch.mean(torch.square(incrs), dim=2)).squeeze()
        Struc[:,1,idx] = torch.mean(torch.pow((incrs-torch.nanmean(incrs))/(torch.sqrt(torch.nanmean(torch.abs(incrs-torch.nanmean(incrs))**2))),3), dim=2).squeeze()
        Struc[:,2,idx] = torch.mean(torch.pow((incrs-torch.nanmean(incrs))/(torch.sqrt(torch.nanmean(torch.abs(incrs-torch.nanmean(incrs))**2))),4), dim=2).squeeze()/3

    return Struc
