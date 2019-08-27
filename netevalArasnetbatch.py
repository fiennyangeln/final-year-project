import numpy as np
import math as mt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
import os    
import copy
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numpy import linalg as LA
import sys
from termcolor import colored, cprint
import pdb;
from utilsArasnet import *



def biasVarianceFc(net,feature,target,nClass,ns,device):
    # for BIAS calculation

    tempEz = feature
    for netLen in range(len(net)): 
        currnet= net[netLen].to(device)
        Eobj   = currnet.eval()
        tempEz = Eobj(tempEz.to(device))
        tempEz = tempEz.detach()
        
        #to get output of conv layers after linearization
        if netLen == ns.no_of_conv_layer:
            ns.meanConvFeatures.update(np.mean(tempEz.cpu().numpy(),axis=0))

        #to get output of first hidden layer given meanConvFeatures as input
        if netLen == ns.no_of_conv_layer+1:
            EY = tempEz
            EY2 = EY**2
        if netLen == (len(net)-2):
            HS = torch.mean(tempEz, dim=0)  # hidden node significance of the last hidden layer

    Ez   = F.softmax(tempEz, dim=1).data
    Bias = torch.mean((Ez-target)**2,dim=0)
    bias = LA.norm(Bias)

    # for VAR calculation
    tempEz = EY2
    for netLen in range(ns.no_of_conv_layer+2,len(net)): # after CNN MLP to the output, just in case CNN in list 0
        currnet= net[netLen].to(device)
        Eobj   = currnet.eval()
        tempEz = Eobj(tempEz.to(device))

    Ez2 = F.softmax(tempEz, dim=1).data
    Var = torch.mean(Ez2-Ez**2,dim=0)
    variance = LA.norm(Var)

    #pdb.set_trace()

    return bias, variance, HS, ns

def evaluationWindow(F_matrix,alpha,cuttingpoint,ns):
    # cutting point: output space for discriminative
    windowSize = 10
    growLayer = 0
    sizeF = len(F_matrix)
    cuttingCandidate = [int(0.25*sizeF),int(0.5*sizeF),int(0.75*sizeF)]
    F_max = np.max(np.array(F_matrix))
    F_min = np.min(np.array(F_matrix))
    miu_F = np.mean(np.array(F_matrix.double()))
    if ns.currTrainingError < ns.prevTrainingError and ns.growFCcounter > 3:
        for cut in cuttingCandidate:
            G_max = np.max(np.array(F_matrix[0:cut]))
            G_min = np.min(np.array(F_matrix[0:cut]))
            miu_G = np.mean(np.array(F_matrix[0:cut].double()))
            
            epsilon_G = (G_max-G_min)*np.sqrt(((cut)/(2*cut*sizeF)*np.log(1/alpha)))
            epsilon_F = (F_max-F_min)*np.sqrt(((sizeF)/(2*cut*sizeF)*np.log(1/alpha)))
        
            print('Grow FC Layer condition ==>')
            condition1 = (epsilon_F + miu_F)
            condition2 = (epsilon_G + miu_G)
            print('condition1 :',condition1,'condition2 : ',condition2)
            if condition1 < condition2:
                print('ALLOWED to grow FC Layer with cutting point :',cut)
                growLayer = 1
                break
    
    if growLayer == 0:
        print('NOT Allowed to grow FCLayer')
    
    return (windowSize, growLayer)

def featureCorrelation(averageFeature):
    return np.average(np.corrcoef(averageFeature))

def growCNNidentification(averageFeature,growCNNLayer,epoch,ns,q):
    # CNN feature correlation
    convCorrelationScore = featureCorrelation(averageFeature)
    ns.empiricalCorrelationScore.update(convCorrelationScore)
    
    if len(q)==10:
        windowCorrAvg = sum(q)/len(q)
        windowCorrStd = np.std(q)
        if windowCorrStd > ns.maxConvStd:
            ns.maxConvStd = windowCorrStd
        if windowCorrStd < ns.minConvStd:
            ns.minConvStd = windowCorrStd
    
    if ns.retrainCounter != 0:
        ns.retrainCounter = ns.retrainCounter - 1
        growCNNLayer = 0

    # reset or update the minimum mean and minimum standard deviation of empiricalCorrelationScore
    if growCNNLayer == 1 or epoch <= 2 or ns.retrainCounter != 0:
        print('reset minimum')
        ns.empiricalCorrelationScore.reset_min()        
    else:
        print('update_minimum')
        ns.empiricalCorrelationScore.update_min()
    
    # calculate the dynamic Threshold for CNN layer growing
    print('convCorrelationScore : ',convCorrelationScore)
    print('epoch > 2 : ',epoch > 2)
    print('ns.retrainCounter : ',ns.retrainCounter)
    print('len(q) : ',len(q))
    if len(q)==10:
        print('LHS < : ',np.absolute(windowCorrAvg-convCorrelationScore))
        print('RHS : ',(convCorrelationScore*ns.minConvStd))   
        print('Cond4 : ',(np.absolute(windowCorrAvg-convCorrelationScore)<(convCorrelationScore*ns.minConvStd)))

    if epoch > 2 and len(q)==10  and ns.retrainCounter == 0 and (np.absolute(windowCorrAvg-convCorrelationScore)<(convCorrelationScore*ns.maxConvStd)):
        growCNNLayer = 1
        ns.retrainCounter = 5
        ns.empiricalCorrelationScore = AverageMeter()
        ns.minConvCorrelationScore = convCorrelationScore
    else:
        growCNNLayer = 0    
    
    return growCNNLayer, convCorrelationScore, ns