#############################################################################
# Code defining methods for updating weights
# (loss functions, regularisation methods and learning rules)
# adapted from https://codeberg.org/mwspratling/HEMLoss/src/branch/main/loss_functions.py
# (c) 2023 Michael William Spratling
#############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_margin_loss(target_ints, y, sepMargin=None, weightPerClass=None, powParam=1.0, reduction='mean', distances=None):
    #(batch x class, batch, int/(batch x class), class, int, string, (class x class))
    #multi-class classification hinge loss. A re-implementation of F.multi_margin_loss
    #to allow experimentation with alternative reduction methods
    # Note: assumes that there is exactly one target class in each sample.
    #set default parameter values
    if sepMargin==None:
        sepMargin=1 # as from paper, default without any margins 
    if weightPerClass==None:
        weightPerClass=1
        
    #assign a weight for each sample in batch
    #weight is equal to the weight associated with the class of that sample
    
    #weightPerSample=torch.sum(weightPerClass*targets, dim=1, keepdim=True)
    if len(target_ints.size())==1:
        target_ints = target_ints.unsqueeze(0)
    
    targets = torch.nn.functional.one_hot(target_ints, 21).float()
    #print("ints")
    #print(target_ints.size())
    #print(y.size())
    #print(distances.size())
    if not (distances is None):
        sepMargin = torch.matmul(targets,distances).detach()
        #(batch x class)

    yTargets=torch.sum(targets*y, dim=1, keepdim=True) # activity of node that represents true class (note using max here would result in -ve activations being replaced by zeros)
    # calc activity of other nodes that is higher than the margin below the activity of node that represents true class
    #error=(1-targets)*F.relu((y-(yTargets-sepMargin))*weightPerSample) # apply weights to scale error (standard method)
#    error=(1-targets)*F.relu((y-(yTargets-sepMargin*weightPerSample))) # apply weights to adjust margin
   #error=(only non-targets)*only_positive(logit_class-logit_target+margin)
    error=(1-targets)*F.relu(y-yTargets+sepMargin)
    if powParam!=1 and reduction!='weightedSum':
        error=F.relu(error)**powParam
    if reduction=='mean':
        loss=torch.mean(error)
    elif reduction=='meanAboveAvg':
        #calculates mean error using only those values above average
        loss=mean_above_average_reduce(error)
    elif reduction=='meanAboveAvgPerClass': # not discussed in paper
        #calculates mean error using only those values above average
        loss=mean_above_average_per_class_reduce(error,targets)
    elif reduction=='none':
        loss=error
    return loss

def mean_above_zero_reduce(error, zeroEq=1e-6):
    #a loss reduction method that only includes non-zero values in the calculation of the mean loss
    #the mean is first calculated separately for each sample, before the mean is calculated over the batch  
    error=torch.sum(error, dim=1)/(zeroEq+torch.sum(error.detach()>0, dim=1))
    loss=torch.sum(error)/(zeroEq+torch.sum(error.detach()>0))
    return loss

def mean_above_average_reduce(error):
    #a loss reduction method that only includes values in the calculation of the mean loss that are above (or equal to) the mean
    #the mean is first calculated separately for each sample, before the mean is calculated over the batch  
    #thres=torch.mean(error.detach())
    #error[error<thres]=0
    #loss=mean_above_zero_all_reduce(error)
    thresPerSample=torch.mean(error.detach(), dim=1, keepdim=True)
    #print(thresPerSample.size())
    #print(error.size())
    #thresPerSample=torch.sum(error.detach(), dim=1, keepdim=True)/(1e-6+torch.sum(error.detach()>0, dim=1, keepdim=True))
    #thresPerSample=torch.median(error.detach(), dim=1, keepdim=True).values
    error=torch.where(error>=thresPerSample.expand(-1, error.size(1), -1),error,torch.zeros_like(error))
    loss=mean_above_zero_reduce(error)
    #the above equivalent to:
    #error=mean_above_average(error, dim=1)
    #loss=mean_above_zero(error)
    return loss

def mean_above_average_per_class_reduce(error,targets): # not discussed in paper
    #a loss reduction method that only includes values in the calculation of the mean loss that are above (or equal to) the mean
    #the mean is first calculated separately for each sample, then for each class, before the mean is calculated over the batch  
    thresPerSample=torch.mean(error.detach(), dim=1, keepdim=True)
    error=torch.where(error>=thresPerSample.expand(-1,error.size(1)),error,torch.zeros_like(error))
    loss=torch.tensor([0.0]).to(error.device)
    for t in range(targets.size(1)):
        idx=targets[:,t]==1
        if torch.sum(idx)>0:
            loss+=mean_above_zero_reduce(error[idx,:])
    loss/=targets.size(1)
    return loss

def mean_non_zero(y, dim=None, keepdim=False, zeroEq=1e-6):
    #find the mean, but exclude values that equal zero from the calculation
    return torch.sum(y, dim=dim, keepdim=keepdim)/(zeroEq+torch.sum(torch.abs(y.detach())>0, dim=dim, keepdim=keepdim))

def mean_above_zero(y, dim=None, keepdim=False, zeroEq=1e-6):
    #find the mean, but exclude values that <= zero from the calculation
    y=F.relu(y)
    return torch.sum(y, dim=dim, keepdim=keepdim)/(zeroEq+torch.sum(y.detach()>0, dim=dim, keepdim=keepdim))

def mean_above_average(y, dim=None, keepdim=False, zeroEq=1e-6):
    #replace below average values with zeros
    thres=torch.mean(y.detach(), dim=dim, keepdim=True)
    y=torch.where(y>=thres.expand(y.size()),y,torch.zeros_like(y))
    return mean_above_zero(y, dim=dim, keepdim=keepdim, zeroEq=zeroEq)
