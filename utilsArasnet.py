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

def get_error( scores , labels ):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator  = (predicted_labels == labels)
    F_matrix   = (predicted_labels != labels)
    num_matches= indicator.sum()
    
    return 1-num_matches.float()/bs, F_matrix, predicted_labels

def one_hot(targets, classes):
    targets = targets.type(torch.LongTensor).view(-1)
    targets_onehot = torch.zeros(targets.size()[0], len(classes))
    for i, t in enumerate(targets):
        if t in classes:
            targets_onehot[i][classes.index(t)] = 1
    return targets_onehot

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
       Modified by Arasnet team
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.avg_old = 0
        self.std = 0
        self.std_old = 0
        self.count = 0
        self.miu_min = 0
        self.std_min = 0

    def update(self, val, n=1):
        self.value   = val
        self.avg_old = copy.deepcopy(self.avg)
        self.std_old = copy.deepcopy(self.std)
        self.count  += n
        self.avg     = self.avg_old + np.divide((val-self.avg_old),self.count)
        self.std     = np.sqrt(self.std_old**2 + self.avg_old**2 - self.avg**2 + ((val**2 - self.std_old**2 - self.avg_old**2)/self.count))
        
    def reset_min(self):
        self.miu_min = copy.deepcopy(self.avg)
        self.std_min = copy.deepcopy(self.std)
        
    def update_min(self):
        if self.avg < self.miu_min:
            self.miu_min = copy.deepcopy(self.avg)
        if self.std < self.std_min:
            self.std_min = copy.deepcopy(self.std)

def get_bias_var( scores , labels ):

    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    
    return 1-num_matches.float()/bs   

def check_cifar_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'cifar/train_data.pt') 
    flag_train_label = os.path.isfile(path_data + 'cifar/train_label.pt') 
    flag_test_data = os.path.isfile(path_data + 'cifar/test_data.pt') 
    flag_test_label = os.path.isfile(path_data + 'cifar/test_label.pt') 
    if flag_train_data==False or flag_train_label==False or flag_test_data==False or flag_test_label==False:
        print('CIFAR dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=True,
                                        download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=False,
                                       download=True, transform=transforms.ToTensor())  
        train_data=torch.Tensor(50000,3,32,32)
        train_label=torch.LongTensor(50000)
        for idx , example in enumerate(trainset):
            train_data[idx]=example[0]
            train_label[idx]=example[1]
        torch.save(train_data,path_data + 'cifar/train_data.pt')
        torch.save(train_label,path_data + 'cifar/train_label.pt') 
        test_data=torch.Tensor(10000,3,32,32)
        test_label=torch.LongTensor(10000)
        for idx , example in enumerate(testset):
            test_data[idx]=example[0]
            test_label[idx]=example[1]
        torch.save(test_data,path_data + 'cifar/test_data.pt')
        torch.save(test_label,path_data + 'cifar/test_label.pt')
    return path_data