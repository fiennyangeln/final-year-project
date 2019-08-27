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
import matplotlib.pyplot as plt

from trainTestArasnetCIFARbatchnorm import test_Cnet,train_Cnet
from utilsArasnet import AverageMeter
from evolveArasnetbatchNorm import createlinearlizationlayer,createhiddenlayer,createoutputlayer,createNewConvLayer, createNewHiddenLayer, createFirstHiddenLayer, deleteFcLayer 
from netevalArasnetbatch import evaluationWindow,featureCorrelation,growCNNidentification

from collections import deque
import pdb
import scipy.io
from termcolor import colored, cprint
from sklearn.model_selection import train_test_split
from collections import deque

class smallconvnet(nn.Module):
    def __init__(self):
        super(smallconvnet, self).__init__()

        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool  = nn.MaxPool2d(2,2)
        
        nn.init.xavier_uniform_(self.conv.weight)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)        
        x = self.pool(x)
        # Droput
        x = self.dropout1(x)
        return x       

def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(nb_param, nb_param/1e6))

## main loop autocnn2
def autocnn2(x_train, y_train, x_val, y_val, x_test, y_test, no_of_epoch, gpu = 'cuda',
	CNNGROW = True, FCGROW = True, NODEGROW = True, NODEPRUNE = True):
    # recursive calculation storagae
    class nsformula:
        pass
    ns = nsformula()
    ns.ky = 0                       # overall counter
    ns.kvar = 0 # counter to wait until netVar stable
    ns.retrainCounter = 0
    trainingMode = 0

    # grow and prune node parameter
    ns.miu_ns = AverageMeter()
    ns.miu_nhs = AverageMeter()

    # grow fully connected layer parameter
    ns.Q = len(y_train)
    ns.no_of_hidden_fcclayer = 0
    ns.alpha_init = 0.05
    ns.alpha = 0.05
    ns.growLayer = 0
    ns.currTrainingError = 0
    ns.prevTrainingError = 0
    fccLayerEvo = []
    ns.growFCcounter = 0
    ns.NODEGROW = NODEGROW
    ns.NODEPRUNE = NODEPRUNE

    # grow fully CNN layer parameter
    ns.meanConvFeatures = AverageMeter()
    ns.no_of_conv_layer = 1
    ns.no_of_pool_layer = 1
    ns.empiricalCorrelationScore = AverageMeter()
    growCNNLayer = 0
    convLayerEvo = []
    countCNNtraining = 0
    nPrevFeatures = 0

    ns.minConvCorrelationScore = 1
    ns.minConvStd = 100
    ns.maxConvStd = 0
    ns.prevTrError = 100
    mode = 1
    stage = 0
    maxTrError = 0
    minTrError = 100
    batch_size = 128

    # prepare model
    #print('\n==> Creating the network')

    ns.size_pool = 2
    ns.input_row = x_train.size(dim=2)
    ns.input_col = x_train.size(dim=3)

    noKernels = 32

    I = int(np.floor(x_train.size(dim=2)/(ns.size_pool)**(ns.no_of_conv_layer))*
            np.floor(x_train.size(dim=3)/(ns.size_pool)**(ns.no_of_conv_layer))*noKernels)

    classes = int(torch.max(y_train)+1)
    
    ArAsNet = [smallconvnet().double(),createlinearlizationlayer().double(), 
               createoutputlayer(I,classes)]
    device = torch.device(gpu)

    # performance
    Error_train = []
    Loss_train  = []

    Error_test = []
    Loss_test  = []

    Error_val = []
    Loss_val = []

    Testing_time  = []
    Training_time = []

    Bias2 = []
    Var   = []
    Iter  = []
    gradBias = deque([])
    netVar = deque([])
    meanGradBias = []

    convFeatureCorr = []
    growCNN = []

    q = deque([])
    ns.trq = deque([])
    ns.trGrad = False
    
    for epoch in range(0,no_of_epoch): 
        torch.cuda.empty_cache()
        print('\n==> Epoch: {}'.format(epoch + 1))
        Iter.append(epoch)

        # pre Test
        error_preTrain, _, _, _, _, _,averageFeature,_ = test_Cnet(ArAsNet, x_train, y_train, 
                                                                   batch_size, device, ns, classes)
        torch.cuda.empty_cache()
        
        #Record min and max training error
        if error_preTrain > maxTrError:
            maxTrError = error_preTrain

        if error_preTrain < minTrError and maxTrError != error_preTrain:
            minTrError = error_preTrain

        # validation
        print("------------IN VALIDATION PHASE------------")
        Error_validation, Loss_validation, _, F_matrix, _, _,_,_ = test_Cnet(ArAsNet, x_val, y_val, 
                                                             batch_size, device, ns, classes)
        torch.cuda.empty_cache()
        
        if epoch == 1:
            ns.prevTrainingError = Error_validation
        ns.currTrainingError = Error_validation
        print('Validation Error: {}'.format(Error_validation))
        print('Validation Loss: {}'.format(Loss_validation))
        Error_val.append(Error_validation)
        Loss_val.append(Loss_validation)
        print("-------------------------------------------")

        # Determine the evaluation window and grow mlp layer signal
        if epoch > 1 and mode == 2:
            chunkSize = len(y_train)
            cuttingpoint = int(len(F_matrix)/2)
            _,ns.growLayer = evaluationWindow(F_matrix,ns.alpha,cuttingpoint,ns)
        else:
            ns.growLayer = 0

        # CNN Layer growing layer identification
        if epoch > 0 and mode == 1:
            growCNNLayer,convCorrelationScore,ns = growCNNidentification(averageFeature,growCNNLayer,epoch,ns,q)
            convFeatureCorr.append(convCorrelationScore)
            if (len(q)<10):
                q.append(convCorrelationScore)
            else:
                q.popleft()
                q.append(convCorrelationScore)
        else:
            _,convCorrelationScore,_ = growCNNidentification(averageFeature,growCNNLayer,epoch,ns,q)
            convFeatureCorr.append(convCorrelationScore)
        
        countCNNtraining += 1
        if growCNNLayer == 1  and countCNNtraining <= 100 and ns.prevTrError > error_preTrain and mode == 1:
        #if growCNNLayer == 1  and countCNNtraining <= 100 and mode == 1:
            nPrevFeatures = ArAsNet[ns.no_of_conv_layer+1].linear.in_features
            if CNNGROW:
                ArAsNet,ns.no_of_conv_layer,ns.no_of_pool_layer = createNewConvLayer(ArAsNet,ns.no_of_conv_layer,
                                                             x_train.size(dim=2),x_train.size(dim=3),ns.size_pool,ns.no_of_pool_layer)
            newLayerIndex       = ns.no_of_conv_layer - 1
            ns.meanConvFeatures = AverageMeter()
            ns.prevTrError      = error_preTrain
            ns.growLayer        = 0
            countCNNtraining    = 0
            print('mode : ',mode,'CNN counter : ',countCNNtraining)
            ns.trq.clear()
            ns.trGrad        = False
            maxTrError       = 0
            minTrError       = 100
        elif countCNNtraining > 10 and ns.prevTrError > error_preTrain and mode == 1:
            growCNNLayer = 0
            print("*******************Minimum Error Threshold REACHED**********************")
            mode = 2
            countCNNtraining = 11
            print('mode : ',mode,'CNN counter : ',countCNNtraining)
            ns.trq.clear()
            ns.trGrad        = False
            maxTrError       = 0
            minTrError       = 100
        elif (countCNNtraining > 10 and ns.prevTrError <= error_preTrain and mode == 1) or (ns.trGrad and mode == 1):
            growCNNLayer = 0
            print("*******************Minimum Error Threshold NOT Reached**********************")
            print("*******************Optimize CNN for 10 epochs: started**********************")
            print("*******************Stopping criteria for CNN growing**********************")
            mode             = 2
            trainingMode     = 0
            ns.trq.clear()
            ns.trGrad        = False
            maxTrError       = 0
            minTrError       = 100
            countCNNtraining = 0
            FCGROW           = False
        growCNN.append(ns.no_of_conv_layer)

        # transition mode
        if mode == 2 and countCNNtraining == 0:
            print("*******************Optimize CNN and FC for 10 epochs: started**********************")
            trainingMode = 6

        if mode == 2 and countCNNtraining == 11:
            print("*******************Optimize FC: started**********************")
            FCGROW       = True
            trainingMode = 7
            ns.trq.clear()
            ns.trGrad    = False
            maxTrError   = 0
            minTrError   = 100

        if mode == 2 and len(netVar) == 10 and len(gradBias) == 10 and ns.prevTrError >= error_preTrain and FCGROW:            
            ns.prevTrError = error_preTrain
            epsilon_d      = np.sqrt((1/(2*10))*np.log(1/ns.alpha))            
            meanGradBias   = np.average(gradBias)
            gradBiasMin    = np.min(gradBias)
            gradBiasMax    = np.max(gradBias)
            meanNetVar     = np.average(netVar)
            gradVar        = (ns.miu_nhs.avg - ns.miu_nhs.avg_old)
            grad_bs        = (ns.miu_ns.avg - ns.miu_ns.avg_old)

            condition1 = np.absolute(meanGradBias-gradBiasMin)
            condition2 = (gradBiasMax - gradBiasMin)*epsilon_d/meanNetVar
            print('condition__1 : ',condition1,'condition__2 : ',condition2)
            if condition1 < condition2:
                if ns.no_of_hidden_fcclayer == 0:
                    # grow the first hidden layer
                    growLayer = 1
                    nFeatures = ArAsNet[ns.no_of_conv_layer+1].linear.in_features
                    ArAsNet = createFirstHiddenLayer(ArAsNet,nFeatures,classes)
                    if NODEGROW == False:
                        ArAsNet = growNode(ArAsNet,50-classes)
                    newLayerIndex = len(ArAsNet)-2
                    ns.no_of_hidden_fcclayer += 1
                    ns.miu_ns = AverageMeter()
                    ns.miu_nhs = AverageMeter()
                    ns.kvar = 0
                    trainingMode = 1
                    ns.growFCcounter = 0
                    ns.trq.clear()
                    netVar.clear()
                    gradBias.clear()
                    ns.trGrad = False
                    maxTrError = 0
                    minTrError = 100
                elif ns.no_of_hidden_fcclayer > 0:
                	# grow hidden layer
                    growLayer = 1
                    ns.prevTrainingError = ns.currTrainingError
                    ns.prevTrError = error_preTrain
                    ArAsNet = createNewHiddenLayer(ArAsNet,classes)
                    if NODEGROW == False:
                        ArAsNet = growNode(ArAsNet,50-classes)
                    newLayerIndex = len(ArAsNet)-2
                    ns.no_of_hidden_fcclayer += 1
                    ns.miu_ns = AverageMeter()
                    ns.miu_nhs = AverageMeter()
                    ns.kvar = 0
                    ns.growFCcounter = 0
                    ns.trq.clear()                
                    netVar.clear()
                    gradBias.clear()
                    ns.trGrad = False
                    maxTrError = 0
                    minTrError = 100

        # performance evaluation window
        if (len(ns.trq) < 10):
            ns.trq.append(error_preTrain)
        elif (len(ns.trq) == 10):
            firstError = ns.trq.popleft()
            lastError = ns.trq.pop()
            if ((firstError - lastError)/(maxTrError-minTrError)) < 0.1*np.sqrt((1/(2*10))*np.log(1/ns.alpha)):
                ns.trGrad = True

        # delete the last FC layer if it wont help to improve the performance           
        if (mode == 2 and ns.trGrad):
            print("*******************Network Evolution: deleting last hidden layer**********************")
            if error_preTrain > ns.prevTrError and ns.no_of_hidden_fcclayer >= 1:
                ArAsNet,ns = deleteFcLayer(ArAsNet,classes,ns)
                # optimize ALL layers
                ns.miu_ns = AverageMeter()
                ns.miu_nhs = AverageMeter()
                ns.kvar = 0
            if stage == 0:
                print("*******************Optimize both CNN and FC together**********************")
                mode = 2
                stage = 3
                trainingMode = 2
                ns.trq.clear()
                ns.trGrad = False
                maxTrError = 0
                minTrError = 100
            else:
                mode = 3
                stage = 4

        # delete the last FC layer if it wont help to improve the performance           
        #if (mode == 2 and ns.trGrad):
        #    if error_preTrain > ns.prevTrError and ns.no_of_hidden_fcclayer > 1:
        #        ArAsNet,ns = deleteFcLayer(ArAsNet,classes,ns)
        #        print("*******************Network Evolution: deleting last hidden layer**********************")
        #        # optimize ALL layers
        #        mode = 3
        #        trainingMode = 2
        #        ns.trq.clear()
        #        ns.trGrad = False
        #        maxTrError = 0
        #        minTrError = 100
        #    else:
        #        print("*******************Network Evolution: without deleting last hidden layer**********************")
        #        # optimize ALL layers
        #        mode = 3
        #        trainingMode = 2
        #        ns.trq.clear()
        #        ns.trGrad = False
        #        maxTrError = 0
        #        minTrError = 100
         
        # end the training process
        if mode == 3 and stage == 4 and ns.trGrad:
            mode = 4
            endCounter = 0
        if mode == 4:
            endCounter += 1
            if endCounter == 30:
                break

        # Train
        start_train = time.time()
        ns.growFCcounter += 1
        ArAsNet, ns = train_Cnet(ArAsNet, x_train, y_train, device, classes, ns, batch_size, trainingMode)
        end_train   = time.time()
        torch.cuda.empty_cache()

        training_time = end_train - start_train
        Training_time.append(training_time)

        # Training result
        print("------------IN TRAINING PHASE------------")
        error_train, loss_train, _, _,_,_,_,_ = test_Cnet(ArAsNet, x_train, y_train, batch_size, device, ns, classes)  
        print('Training Error: {}'.format(error_train))
        print('Training Loss: {}'.format(loss_train))
        print('Training Time: {}'.format(training_time))
        print("-------------------------------------------")
        torch.cuda.empty_cache()
        
        Error_train.append(error_train)
        Loss_train.append(Loss_train)

        # Testing result
        print("------------IN Testing PHASE---------------")
        start_test = time.time()
        error_test, loss_test, _, _,_,_,_,conf_mat = test_Cnet(ArAsNet, x_test, y_test, batch_size, device, ns, classes)
        print('Testing Error: {}'.format(error_test))
        print('Testing Loss: {}'.format(loss_test))
        print('Confussion Matrix: {}'.format(conf_mat))
        torch.cuda.empty_cache()
        
        end_test     = time.time()
        testing_time = end_test - start_test
        Testing_time.append(training_time)
        print('Testing Time: {}'.format(testing_time))

        Error_test.append(error_test)
        Loss_test.append(loss_test)

        print("-------------------------------------------")

        Bias2.append(ns.miu_ns.avg)
        Var.append(ns.miu_nhs.avg)
        
        #calculate gradient bias and variance
        if (len(gradBias) < 10):
            grad_bs = (ns.miu_ns.avg - ns.miu_ns.avg_old)
            gradBias.append(grad_bs)
        elif (len(gradBias) == 10):
            winFirstGBias = gradBias.popleft()
            gradBias.append(grad_bs)

        if (len(netVar) < 10):
            netVar.append(ns.miu_nhs.avg)
        elif (len(netVar) == 10):
            winFirstVar = netVar.popleft()
            netVar.append(ns.miu_nhs.avg)
    
        fccLayerEvo.append(ns.no_of_hidden_fcclayer)
        convLayerEvo.append(ns.no_of_conv_layer)

    print("\n==> Results")

    for netLen in range(len(ArAsNet)):
        display_num_param(ArAsNet[netLen])

    print('Testing time mean: {}'.format(np.mean(Testing_time)))
    print('Testing time std: {}'.format(np.std(Testing_time)))
    print('Training time mean: {}'.format(np.mean(Training_time)))
    print('Training time std: {}'.format(np.std(Training_time)))
    
    return (ArAsNet, fccLayerEvo, convLayerEvo, Bias2, Var, convFeatureCorr, Error_test, Error_val, Error_train,
           Loss_test, Loss_val, Loss_train, Iter)