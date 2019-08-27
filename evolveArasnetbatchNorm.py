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
import pdb

class hiddenlayer(nn.Module):
    def __init__(self, no_input, no_output):
        super(hiddenlayer, self).__init__()
        self.linear = nn.Linear(no_input, no_output)
        self.bn = nn.BatchNorm1d(no_output)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.linear.weight)
        self.dropout1 = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        # Droput
        x = self.dropout1(x)
        return x
    
def createhiddenlayer(no_input,no_output):
    obj = hiddenlayer(no_input,no_output).double()
    return obj

class outputlayer(nn.Module):
    def __init__(self, no_input, classes):
        super(outputlayer, self).__init__()
        self.linear = nn.Linear(no_input, classes)
        self.bn = nn.BatchNorm1d(classes)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.linear.weight)
        self.dropout1 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        # Droput
        x = self.dropout1(x)
        return x
    
def createoutputlayer(no_input,classes):
    obj = outputlayer(no_input,classes).double()
    return obj

class convolutionlayer(nn.Module):
    def __init__(self, noChannels, noKernels, kernelSize):
        super(convolutionlayer, self).__init__()        
        self.conv = nn.Conv2d(noChannels, noKernels, kernel_size=kernelSize, padding=int(np.floor(kernelSize[0]/2)))
        self.bn = nn.BatchNorm2d(noKernels)
        self.relu = nn.ReLU()
        self.pool  = nn.MaxPool2d(2,2)
        nn.init.xavier_uniform_(self.conv.weight)
        self.dropout1 = nn.Dropout(0.3) 
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        # Droput
        x = self.dropout1(x)
        return x
    
def createconvolutionlayer(noChannels, noKernels, kernelSize):
    obj = convolutionlayer(noChannels, noKernels, kernelSize).double()
    
    return obj

class convolutionlayernopool(nn.Module):
    def __init__(self,noChannels, noKernels, kernelSize):
        super(convolutionlayernopool, self).__init__()        
        self.conv = nn.Conv2d(noChannels, noKernels, kernel_size=kernelSize, padding=int(np.floor(kernelSize[0]/2)))
        self.bn = nn.BatchNorm2d(noKernels)
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.conv.weight)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def createconvolutionlayernopool(noChannels, noKernels, kernelSize):
    obj = convolutionlayernopool(noChannels, noKernels, kernelSize).double()
    
    return obj

class linearlizationlayer(nn.Module):
    def __init__(self):
        super(linearlizationlayer, self).__init__()
        
    def forward(self, x):
        d1,d2,d3,d4 = x.size()        
        x = x.view(-1, d2*d3*d4)
        
        return x
    
def createlinearlizationlayer():
    obj = linearlizationlayer().double()
    
    return obj

def deleteFcLayer(net,classes,ns):
    net = np.delete(net, len(net)-2)
    net = np.delete(net, len(net)-1)
    if ns.no_of_hidden_fcclayer == 1:
        noKernels = net[ns.no_of_conv_layer-1].conv.out_channels
        nFeatures = int(np.floor(ns.input_row/(ns.size_pool)**(ns.no_of_pool_layer+1))*np.floor(ns.input_col/(ns.size_pool)**(ns.no_of_pool_layer+1))*noKernels)
    else:
        nFeatures = net[len(net)-1].linear.out_features
    newOutput = createoutputlayer(nFeatures,classes)
    net = np.insert(net, len(net), newOutput)
    ns.no_of_hidden_fcclayer -= 1
    text = colored('A new FC layer is deleted', 'blue', attrs=['reverse', 'blink'])
    print(text) 

    return net,ns

def createNewConvLayer(net,no_of_conv_layer,input_row,input_col,size_pool,no_of_pool_layer):    
    cprint("A new CNN layer is created", 'red', attrs=['bold'], file=sys.stderr)
    noChannels = net[no_of_conv_layer-1].conv.out_channels
    noKernels = net[no_of_conv_layer-1].conv.out_channels
    kernelSize = net[no_of_conv_layer-1].conv.kernel_size
    idxFirstHiddenLayer = no_of_conv_layer+1

    if no_of_conv_layer%3 == 0 and noKernels<256:
        noKernels = noKernels*2
    
    n_in = int(np.floor(input_row/(size_pool)**(no_of_pool_layer+1))*np.floor(input_col/(size_pool)**(no_of_pool_layer+1))*noKernels)

    print("dimension : ",n_in)

    sizeOfFirstHiddenLayer = net[idxFirstHiddenLayer].linear.out_features
    storeCNNFeatureWeights = copy.deepcopy(net[idxFirstHiddenLayer].linear.weight.data)
    

    #newNet = np.copy(np.insert(net, no_of_conv_layer, createconvolutionlayernopool(noChannels,noKernels,kernelSize)))
    #n_in = int(np.floor(input_row/(size_pool)**(no_of_pool_layer))*np.floor(input_col/(size_pool)**(no_of_pool_layer))*noKernels)

    if n_in > 2000:        
        newNet = np.copy(np.insert(net, no_of_conv_layer, createconvolutionlayer(noChannels,noKernels,kernelSize)))
        no_of_pool_layer = no_of_pool_layer + 1
    else:
        newNet = np.copy(np.insert(net, no_of_conv_layer, createconvolutionlayernopool(noChannels,noKernels,kernelSize)))
        n_in = int(np.floor(input_row/(size_pool)**(no_of_pool_layer))*np.floor(input_col/(size_pool)**(no_of_pool_layer))*noKernels)
    
    idxFirstHiddenLayer = idxFirstHiddenLayer+1        
    newNet[idxFirstHiddenLayer] = createoutputlayer(n_in,sizeOfFirstHiddenLayer)
    
    return newNet,no_of_conv_layer+1,no_of_pool_layer

def oldcreateNewHiddenLayer(net,hiddenSize):
    text = colored('A new FCC layer is created', 'red', attrs=['reverse', 'blink'])
    print(text) 
    n_in    = net[len(net)-2].linear.out_features #18
    classes = net[len(net)-1].linear.out_features #2
    
    newNet = np.copy(np.insert(net, len(net)-1, createhiddenlayer(n_in,int(n_in/2))))
    
    storeOutputWeights = np.delete(copy.deepcopy(net[len(net)-1].linear.weight.data), np.arange(int(n_in/2), n_in+1), 1)

    newNet = np.copy(newNet[:-1])
    newNet = np.append(newNet,createoutputlayer(int(n_in/2),classes))
    newNet[len(newNet)-1].linear.weight.data = copy.deepcopy(storeOutputWeights)
    
    return newNet

def createNewHiddenLayer(net,hiddenSize):
    text = colored('A new FCC layer is created', 'red', attrs=['reverse', 'blink'])
    print(text) 
    n_in    = net[len(net)-2].linear.out_features #curr hidden layer dimension
    classes = net[len(net)-1].linear.out_features #output layer dimension
    hiddenSize = int(0.25*n_in)
    newNet = np.copy(np.insert(net, len(net)-1, createhiddenlayer(n_in,hiddenSize)))

    #copy old output weights truncated to new hidden size
    currOutputWeights = copy.deepcopy(net[len(net)-1].linear.weight.data)
    newHiddenLayerWeights = copy.deepcopy(newNet[len(newNet)-2].linear.weight.data)
    newHiddenLayerWeights[0:classes,:]=currOutputWeights

    newNet = np.copy(newNet[:-1])
    newNet = np.append(newNet,createoutputlayer(hiddenSize,classes))

    newNet[len(newNet)-2].linear.weight.data = copy.deepcopy(newHiddenLayerWeights)
    
    return newNet

def createFirstHiddenLayer(net,nFeatures,hiddenSize):
    text = colored('The first FC hidden layer is created', 'red', attrs=['reverse', 'blink'])
    print(text)
    n_in    = nFeatures #18
    classes = net[len(net)-1].linear.out_features #2
    hiddenSize = int(0.25*n_in)
    
    #add new hidden layer just before the output layer
    newNet = np.copy(np.insert(net, len(net)-1, createhiddenlayer(n_in,hiddenSize)))
    
    #copy old output weights truncated to new hidden size
    currOutputWeights = copy.deepcopy(net[len(net)-1].linear.weight.data)
    newHiddenLayerWeights = copy.deepcopy(newNet[len(newNet)-2].linear.weight.data)
    newHiddenLayerWeights[0:classes,:]=currOutputWeights

    #delete last output layer
    newNet = np.copy(newNet[:-1])

    #create new output layer 
    newNet = np.append(newNet,createoutputlayer(hiddenSize,classes))
    
    #initialize the weights of the first hidden layer with output weights
    newNet[len(newNet)-2].linear.weight.data = copy.deepcopy(newHiddenLayerWeights)
    
    return newNet

def oldcreateFirstHiddenLayer(net,nFeatures,hiddenSize):
    text = colored('The first FC hidden layer is created', 'red', attrs=['reverse', 'blink'])
    print(text)
    n_in    = nFeatures #18
    classes = net[len(net)-1].linear.out_features #2
    
    newNet = np.copy(np.insert(net, len(net)-1, createhiddenlayer(n_in,hiddenSize)))
    
    storeOutputWeights = np.delete(copy.deepcopy(net[len(net)-1].linear.weight.data), np.arange(hiddenSize, n_in+1), 1)

    newNet = np.copy(newNet[:-1])
    newNet = np.append(newNet,createoutputlayer(hiddenSize,classes))
    newNet[len(newNet)-1].linear.weight.data = copy.deepcopy(storeOutputWeights)
    
    return newNet

def accommodateModel(ArAsNet):
    netOptim = []
    
    for netLen in range(len(ArAsNet)):
        netOptim  = netOptim + list(ArAsNet[netLen].parameters())
        
    return netOptim

def accommodateFCLayer(ArAsNet,ns):
    netOptim = []
    for netLen in range(ns.no_of_conv_layer+1,len(ArAsNet)):
        netOptim  = netOptim + list(ArAsNet[netLen].parameters())
        
    return netOptim

def accommodateCNNLayer(ArAsNet,ns):
    netOptim = []
    for netLen in range(0,ns.no_of_conv_layer):
        netOptim  = netOptim + list(ArAsNet[netLen].parameters())
        
    return netOptim

def accommodateNewModel(ArAsNet,newLayerIndex):
    netOptim = list(ArAsNet[newLayerIndex].parameters())
        
    return netOptim

def growNode(net,nCluster,device):
    nLayer = len(net)
    winninglayer = nLayer - 2
    outputlayer = nLayer - 1
    
    
    [nNode,nCol] = net[winninglayer].linear.weight.data.shape
    nInput = net[winninglayer].linear.in_features
    nOutCurrentLayer = net[winninglayer].linear.out_features
    nNode += nCluster
    nClass = net[outputlayer].linear.out_features
    
    W = copy.deepcopy(net[winninglayer].linear.weight.data)
    b = copy.deepcopy(net[winninglayer].linear.bias.data)
    W_out = copy.deepcopy(net[outputlayer].linear.weight.data)
    
    new_weight = np.sqrt(2/(nInput+1))*torch.rand(nCluster, nInput,dtype=torch.float64)

    new_bias = np.sqrt(2/(nInput+1))*torch.rand(nCluster,dtype=torch.float64)
    new_output_weight = np.sqrt(2/(nOutCurrentLayer+1))*torch.rand(nClass, nCluster,dtype=torch.float64)
    
    W = torch.cat((W,new_weight.to(device)), 0)
    b = torch.cat((b,new_bias.to(device)), 0)
    W_out = torch.cat((W_out,new_output_weight.to(device)), 1)
    
    net[winninglayer].linear.out_features += nCluster
    net[outputlayer].linear.in_features += nCluster
    
    L1 = []
    L2 = []
    n_inNew = net[winninglayer].linear.in_features
    n_outNew = net[winninglayer].linear.out_features
    
    L1 = createhiddenlayer(n_inNew,n_outNew)
    L2 = createoutputlayer(n_outNew,nClass)
    L1.linear.weight.data = copy.deepcopy(W)
    L1.linear.bias.data   = copy.deepcopy(b)
    L2.linear.weight.data = copy.deepcopy(W_out)
    L2.linear.bias.data   = copy.deepcopy(net[outputlayer].linear.bias.data)
    
    net[winninglayer]     = copy.deepcopy(L1)
    net[outputlayer]      = copy.deepcopy(L2)
    
    return net


def pruneNode(net,biasHiddenLayer,device):
    nLayer = len(net)
    winninglayer = nLayer - 2
    outputlayer = nLayer - 1
    
    W = copy.deepcopy(net[winninglayer].linear.weight.data)
    b = copy.deepcopy(net[winninglayer].linear.bias.data)
    W_out = copy.deepcopy(net[outputlayer].linear.weight.data)
    nClass = net[outputlayer].linear.out_features
    
    biasHiddenLayer = biasHiddenLayer.detach()
    biasHiddenLayer /= torch.max(biasHiddenLayer)
    mean_biasHiddenLayer = torch.mean(biasHiddenLayer)
    std_biasHiddenLayer = torch.std(biasHiddenLayer)
    #pdb.set_trace()
    if len(np.argwhere(biasHiddenLayer < np.abs(mean_biasHiddenLayer - std_biasHiddenLayer).to(device))) > 1:
        prune_list = np.argwhere(biasHiddenLayer < np.abs(mean_biasHiddenLayer - std_biasHiddenLayer))[1]
        nPrune = len(prune_list)
    else:
        prune_list = torch.tensor([])
        nPrune = 0
        
    W = np.delete(W, prune_list, 0)
    b = np.delete(b, prune_list, 0)
    W_out = np.delete(W_out, prune_list, 1)
    
    net[winninglayer].linear.out_features -= nPrune
    net[outputlayer].linear.in_features -= nPrune
    
    L1 = []
    L2 = []
    n_inNew = net[winninglayer].linear.in_features
    n_outNew = net[winninglayer].linear.out_features
    
    L1 = createhiddenlayer(n_inNew,n_outNew)
    L2 = createoutputlayer(n_outNew,nClass)
    L1.linear.weight.data = copy.deepcopy(W)
    L1.linear.bias.data   = copy.deepcopy(b)
    L2.linear.weight.data = copy.deepcopy(W_out)
    L2.linear.bias.data   = copy.deepcopy(net[outputlayer].linear.bias.data)
    
    net[winninglayer]     = copy.deepcopy(L1)
    net[outputlayer]      = copy.deepcopy(L2)
    
    return net, nPrune, prune_list