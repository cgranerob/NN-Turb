#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 23:14:12 2022

@author: administrateur
"""

import torch

def analyseIncrsTorchcuda(signal, scales, device="cpu"):
    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis. Fast version (not used when training the model).
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

def analyseIncrsTorchcuda_vp(signal,scales, device='cpu'):

    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis. Slow (ancient) version, used when training the model
    '''  
    
    Nreal=signal.size()[0]
    Struc=torch.zeros((Nreal,3,len(scales)), dtype=torch.float32, device=device)
        
    for ir in range(Nreal):
        
        # We normalize the image by centering and standarizing it
        nanstdtmp=torch.sqrt(torch.nanmean(torch.abs(signal[ir]-torch.nanmean(signal[ir]))**2))
        tmp=(signal[ir]-torch.nanmean(signal[ir]))/nanstdtmp   

        for isc in range(len(scales)):
                
            scale=int(scales[isc])
                
            incrs=tmp[0,scale:]-tmp[0,:-scale]
            incrs=incrs[~torch.isnan(incrs)]
            Struc[ir,0,isc]=torch.log(torch.nanmean(incrs.flatten()**2))
            nanstdincrs=torch.sqrt(torch.nanmean(torch.abs(incrs-torch.nanmean(incrs))**2))
            incrsnorm=(incrs-torch.nanmean(incrs))/nanstdincrs
            Struc[ir,1,isc]=torch.nanmean(incrsnorm.flatten()**3)
            Struc[ir,2,isc]=torch.nanmean(incrsnorm.flatten()**4)/3
        
    return Struc
