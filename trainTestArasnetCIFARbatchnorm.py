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
from sklearn.metrics import confusion_matrix

from utilsArasnet import *
from evolveArasnetbatchNorm import *
from netevalArasnetbatch import *

import pdb;

def test_Cnet(netlist, test_data, test_label, batch_size, device, ns, nClass, sampleSelection = False):
    running_error = 0
    running_loss  = 0
    num_batches   = 0
    F_matrix = []
    deletedIndices = []
    convFeatures = []
    convLabel = []
    averageFeature = []
    predicted_labels = []
    true_labels = []
    criterion = nn.CrossEntropyLoss()

    teS = len(test_data)
    shuffled_indices = torch.randperm(teS)
    
    start_test = time.time()
    for i in range(0, teS, batch_size):
        indices         = shuffled_indices[i:i+batch_size]
        minibatch_data  = test_data [indices]
        minibatch_label = test_label[indices]
        minibatch_data  = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)
        inputs = minibatch_data
        tempVar = inputs
        for netLen in range(len(netlist)):
            currnet = netlist[netLen].to(device)
            obj = currnet.eval();
            tempVar = obj(tempVar)
            if netLen == (ns.no_of_conv_layer):
                convLabel = convLabel +  [ int(x) for x in minibatch_label ]
                convFeatures.append(tempVar.detach())
        tempVar = tempVar.detach()
        scores = tempVar
       
        # performance calculation
        minibatch_label = minibatch_label.long()
        loss = criterion(scores,minibatch_label)

        running_loss += loss.detach().item()
        error, F, pred_labels = get_error(scores, minibatch_label)
        predicted_labels = predicted_labels +  [ int(x) for x in pred_labels ]
        true_labels = true_labels +  [ int(x) for x in minibatch_label ]
        running_error += error.item()
        F_matrix.append(F)
        num_batches += 1
    
    end_test = time.time()
    F_matrix = torch.cat(F_matrix,0)
    
    # calculate average class feature
    torch.cuda.empty_cache()
    #pdb.set_trace()
    convFeatures = torch.cat(convFeatures,0)
    convFeatures = convFeatures.to('cpu')
    convFeaturesArr = np.array(convFeatures)
    for iClass in range(0,nClass):
        averageFeature.append(np.mean(convFeaturesArr[np.array(convLabel)==iClass], axis=0).squeeze())

    # calculate performance
    testing_time = end_test - start_test
    total_error = running_error/num_batches
    total_loss = running_loss/num_batches
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    
    torch.cuda.empty_cache()

    return total_error*100, total_loss, testing_time, F_matrix, test_data, test_label, averageFeature, conf_mat


def train_Cnet(ArAsNet, train_data, train_label, device, classes, ns, batch_size, trainingMode, shouldGrowNode  = True,
    shouldPruneNode = True):
    # hidden node growing and pruning variable
    grow  = 0
    prune = 0
    
    # lerning variable
    lr = 0.001 #learning rate
    criterion = nn.CrossEntropyLoss()
    
    # Determine training mode
    if trainingMode == 0:
        netOptim = accommodateCNNLayer(ArAsNet,ns)
        optimizer = torch.optim.Adam(netOptim, lr=lr)
        shouldGrowNode = False
        shouldPruneNode = False
        print('Accomodate CNN layers')
    elif trainingMode == 1:
        netOptim = accommodateFCLayer(ArAsNet,ns)
        optimizer = torch.optim.Adam(netOptim, lr=lr)
        print('Accomodate FC layers')
    elif trainingMode == 2:
        netOptim = accommodateModel(ArAsNet)
        optimizer = torch.optim.Adam(netOptim, lr=lr)
        print('Accomodate ALL layers')
    elif trainingMode == 3:
        netOptim = accommodateModel(ArAsNet)
        optimizer = torch.optim.Adam(netOptim, lr = 0.1*lr)
        shouldGrowNode = False
        shouldPruneNode = False
        print('Accomodate ALL layers without evolution smaller learning rate')
    elif trainingMode == 4:
        netOptim = accommodateCNNLayer(ArAsNet,ns)
        optimizer = torch.optim.SGD(netOptim, lr=0.01)
        print('Accomodate CNN layers: SGD')
    elif trainingMode == 5:
        netOptim = accommodateModel(ArAsNet)
        optimizer = torch.optim.SGD(netOptim, lr=0.01)
        print('Accomodate ALL layers: SGD')
    elif trainingMode == 6:
        netOptim = accommodateModel(ArAsNet)
        optimizer = torch.optim.Adam(netOptim, lr=lr)
        print('Accomodate ALL layers without evolution')
        shouldGrowNode = False
        shouldPruneNode = False
    elif trainingMode == 7:
        netOptim = accommodateFCLayer(ArAsNet,ns)
        optimizer = torch.optim.Adam(netOptim, lr=lr)
        print('Accomodate FC layers without evolution')
        shouldGrowNode = False
        shouldPruneNode = False

    # growing and pruning mode
    if ns.NODEGROW == False:
        shouldGrowNode = False
    if ns.NODEPRUNE == False:
        shouldPruneNode = False
    
    # data preparation
    trN = len(train_data)
    shuffled_indices = torch.randperm(trN)

    for count in range(0,trN,batch_size):
        # Set the gradients to zeros
        optimizer.zero_grad()

        if ns.ky%5000 == 0:
            print('=====**** Data point : ', ns.ky)
        
        growLayer = 0

        # Prepare input
        indices         = shuffled_indices[count:count+batch_size]
        minibatch_data  = train_data[indices]
        minibatch_label = train_label[indices]
        minibatch_data  = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)
        inputs          = minibatch_data

        # bias variance calculation===========================================================================================
        z = torch.tensor(one_hot(minibatch_label, range(classes)), dtype=torch.float64) # true class label
        NS,NHS,HS,ns = biasVarianceFc(ArAsNet,inputs,z.to(device),classes,ns,device)
        #pdb.set_trace()
        
        # calculate empirical mean and standard deviation of bias and variance
        ns.miu_ns.update(NS)
        ns.miu_nhs.update(NHS)
        
        # reset or update the minimum mean and minimum standard deviation of bias and variance
        if grow == 1 or ns.kvar <= 1:
            ns.miu_ns.reset_min()
        else:
            ns.miu_ns.update_min()
        if ns.kvar <= 10 or prune == 1:
            ns.miu_nhs.reset_min()
        else:
            ns.miu_nhs.update_min()
        
        # calculate the dynamic Threshold for hidden node growing
        miustd_NS    = ns.miu_ns.avg + ns.miu_ns.std
        miustdmin_NS = ns.miu_ns.miu_min + (1.5*np.exp(-NS)+0.5)*ns.miu_ns.std_min
        if ns.ky%5000 == 0:
            print('miustd_NS : ',miustd_NS,'| miustdmin_NS : ',miustdmin_NS)
        
        # hidden node growing
        if growLayer != 1 and miustd_NS >= miustdmin_NS and shouldGrowNode and ns.no_of_hidden_fcclayer >= 1:
            grow = 1
            nCluster = 64
            ArAsNet = growNode(ArAsNet,nCluster,device)
            print(colored('There are : ', 'green'),nCluster, colored('nodes ADDED around sample', 'green'),ns.ky)
            print('Weight size : ',ArAsNet[len(ArAsNet)-2].linear.weight.shape, '| Out features : ',
                  ArAsNet[len(ArAsNet)-2].linear.out_features)
        else:
            grow = 0
        
        # ===========================================================================================================

        # calculate the dynamic Threshold for hidden node growing
        miustd_NHS    = ns.miu_nhs.avg + ns.miu_nhs.std
        miustdmin_NHS = ns.miu_nhs.miu_min + 2*(1.5*np.exp(-NHS)+0.5)*ns.miu_nhs.std_min
        if ns.ky%5000 == 0:
            print('miustd_NHS : ',miustd_NHS,'| miustdmin_NHS : ',miustdmin_NHS)
        
        if (growLayer != 1 and grow == 0 and miustd_NHS >= miustdmin_NHS and ns.kvar > 20 and shouldPruneNode
            and ns.no_of_hidden_fcclayer >= 1):
            if ArAsNet[len(ArAsNet)-2].linear.out_features > 1 :
                prune = 1
                biasHiddenLayer = HS
                ArAsNet, nPrune, prunedNode = pruneNode(ArAsNet,biasHiddenLayer,device)
                if nPrune > 0:
                    print(colored('There are : ', 'blue'),nPrune, colored('nodes PRUNED around sample', 'blue'),ns.ky)
                    print('Pruned Nodes Number: ',prunedNode)
                    print('Weight size after prune : ',
                        ArAsNet[len(ArAsNet)-2].linear.weight.shape,
                         '| Out features : ',ArAsNet[len(ArAsNet)-2].linear.out_features)
        else:
            prune = 0
        
        # ===========================================================================================================
        ns.ky += 1
        ns.kvar += 1
        
        #Define optimizer
        if grow == 1 or prune == 1:
            # Determine training mode
            if trainingMode == 0:
                netOptim = accommodateCNNLayer(ArAsNet,ns)
                optimizer = torch.optim.Adam(netOptim, lr=lr)
            elif trainingMode == 1:
                netOptim = accommodateFCLayer(ArAsNet,ns)
                optimizer = torch.optim.Adam(netOptim, lr=lr)
            elif trainingMode == 2:
                netOptim = accommodateModel(ArAsNet)
                optimizer = torch.optim.Adam(netOptim, lr=lr)
            elif trainingMode == 4:
                netOptim = accommodateCNNLayer(ArAsNet,ns)
                optimizer = torch.optim.SGD(netOptim, lr=0.01)
            elif trainingMode == 5:
                netOptim = accommodateModel(ArAsNet)
                optimizer = torch.optim.SGD(netOptim, lr=0.01)
        
        tempVar = inputs
        tempVar.requires_grad_()
        
        for netLen in range(len(ArAsNet)):
            currnet   = ArAsNet[netLen].to(device)
            obj       = currnet.train()
            tempVar   = obj(tempVar)
        scores = tempVar
        
        # calculate loss
        minibatch_label = minibatch_label.long()
        loss            = criterion(scores,minibatch_label)            

        # update the parameters
        loss.backward()                    
        optimizer.step()

    
    return (ArAsNet, ns)