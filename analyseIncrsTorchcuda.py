#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 23:14:12 2022

@author: administrateur
"""

import torch

def analyseIncrsTorchcuda(signal,scales, device='cpu'):

    '''
    signal is the signal of study and scales is an array with the values of the scales of analysis
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

