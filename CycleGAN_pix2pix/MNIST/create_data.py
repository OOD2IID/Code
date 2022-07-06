# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 17:01:15 2021

@author: abder
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from easydict import EasyDict
from train_mnist import PyNet
import numpy as np
import os

from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method

# Get current working directory
cwd = os.getcwd()
attack='PGD'
# load cifar10
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
'''


def save_img(x,path):
    
    
    x=x.cpu()
    x=x.detach().numpy()
    x=x.reshape((28,28))
    plt.imshow(x, cmap='gray', interpolation='none')
    plt.savefig(path)
    
    
batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('.MNIST', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('.MNIST', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

report = EasyDict(nb_test=0, correct=0)
#for batch_idx, (data, target) in enumerate(train_loader):
for batch_idx, (data, target) in enumerate(test_loader):
    #print('saving target training batch',batch_idx)
    #for i in range(batch_size_train):
    #    save_img(data[i],'./trainB/img{}_{}'.format(i+(batch_idx)*batch_size_train, target[i]))
        
    
    print(data.shape)
    clf = PyNet()
    clf.cuda()
    clf.load_state_dict(torch.load(os.path.join(cwd,'CNN_MNIST.pth')))
    data=data.cuda()
    if attack=='FGS':
        data = fast_gradient_method(clf, data, eps=0.3, norm = np.inf)
    if attack=='PGD':
        data = projected_gradient_descent(clf, data, 0.3, 0.01, 40, np.inf)
    clf.eval()
    _, y_pred = clf(data).max(1)
    report.nb_test += target.size(0)
    target=target.cuda()
    report.correct += y_pred.eq(target).sum().item()
    print('Current accuracy Under {} is {}'.format(attack,report.correct / report.nb_test * 100.0))
    
    print('saving source training batch',batch_idx)
    for i in range(batch_size_test):
        save_img(data[i],'./testA/img{}_{}'.format(i+(batch_idx)*batch_size_test, target[i]))
    
    #if (batch_idx+1)*batch_size_test == 5000:
    #    break
    
    break
    



