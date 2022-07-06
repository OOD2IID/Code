# --coding:utf-8--
'''
@author: cailikun
@time: 19-3-25 下午4:43
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, transforms

from absl import app, flags
from easydict import EasyDict
import numpy as np
import torchvision
from torchvision.datasets import MNIST
import torch.optim as optim

import os
# Get current working directory
cwd = os.getcwd()

class PyNet(nn.Module):
    """CNN architecture. This is the same MNIST model from pytorch/examples/mnist repository"""

    def __init__(self, in_channels=1):
        super(PyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def ld_mnist(root, batch_size=128, transform=None,shuffle=True):
    """Load training and test data."""
    
    if transform==None:
        train_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))
             ]
        )
        test_transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
             ]
        )
    else:
        train_transforms = transform
        
        test_transforms = transform

    # Load MNIST dataset
    train_dataset = MNIST(root=root, train=True, download=True, transform=train_transforms)
    test_dataset = MNIST(root=root, train=False, download=True, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    #print(len(test_loader))
    return EasyDict(train=train_loader, test=test_loader)

class modelA(nn.Module):
    def __init__(self):
        super(modelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 20 * 20, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class modelB(nn.Module):
    def __init__(self):
        super(modelB, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.conv2 = nn.Conv2d(64, 128, 6)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 12 * 12, 10)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class modelC(nn.Module):
    def __init__(self):
        super(modelC, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class modelD(nn.Module):
    def __init__(self):
        super(modelD, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 300)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(300, 300)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(300, 300)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(300, 300)
        self.dropout4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x

def model_mnist(type=1):
    '''
    Defines MNIST model
    '''
    models = [modelA, modelB, modelC, modelD]
    return models[type]()

def load_model(model_path, type=1):
    model = model_mnist(type=type)
    model.load_state_dict(torch.load(model_path+'.pkl'))
    return model
