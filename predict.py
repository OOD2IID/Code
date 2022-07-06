# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
import os
import pickle
import dill
import numpy as np
import argparse
import data
import time
import logging
from collections import OrderedDict
import faiss
from PIL import Image
import matplotlib.pyplot as plt
from utils import load_M, tensor2im, save_image

import torch
import torch.nn as nn
import torchvision
#from torchvision.utils import save_image
from tqdm import tqdm
from MNIST.train_mnist import PyNet, ld_mnist
from CIFAR10.pytorch.train_cifar10 import ld_cifar10
from CIFAR10.pytorch.models import *
from CIFAR10.pytorch.utils import progress_bar

from easydict import EasyDict
from random import random
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#import torchvision.datasets as datasets
#import torchvision.transforms as transforms
from torchvision import transforms
import torchvision.models as models
from robustness1.robustness import model_utils
from robustness1.robustness import datasets as rob_datasets
#from cifar import get_datasets, CNN



from CycleGAN_pix2pix.options.test_options import TestOptions
from CycleGAN_pix2pix.models import create_model
from CycleGAN_pix2pix.data import create_dataset
#from CycleGAN_pix2pix.util.visualizer import save_images


from SSD.models import SupResNet, SSLResNet
from SSD.utils import (
    get_features,
    get_features_1batch,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
    knn,
)


# Get current working directory
cwd = os.getcwd()
opt = TestOptions().parse()
# create execution device
if torch.cuda.is_available():
    if len(opt.gpu_ids) == 1:# and opt.dataset != 'CIFAR10':
        device = 'cuda:'+str(opt.gpu_ids[0])
    else:
        device = 'cuda'
else:
    device = 'cpu'

def init_SSD(train_loader, test_loader,opt):
    
    '''OOD detection'''
    
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    # create model
    #if args.training_mode in ["SimCLR", "SupCon"]:
    

    # load checkpoint
    if opt.dataset == 'MNIST':
        SSD_model = SSLResNet(arch='resnet18',in_channel=1).eval()
        if device =='cuda':
            SSD_model.encoder = nn.DataParallel(SSD_model.encoder).to(device)
        else:
            SSD_model.encoder = SSD_model.encoder.to(device)
        ckpt_dict = torch.load(os.path.join(cwd,'SSD','train_MNIST','latest_exp','checkpoint','model_best.pth.tar'), map_location="cpu")
    elif opt.dataset == 'CIFAR10' or opt.dataset == 'IMAGENET2CIFAR10' or opt.dataset == 'dark2clear':
        SSD_model = SSLResNet(arch='resnet50').eval()
        if device =='cuda':
            SSD_model.encoder = nn.DataParallel(SSD_model.encoder).to(device)
        else:
            SSD_model.encoder = SSD_model.encoder.to(device)
        ckpt_dict = torch.load(os.path.join(cwd,'SSD','CIFAR10.pth'), map_location="cpu")
        
    elif opt.dataset == 'IMAGENET' or opt.dataset == 'sharp2normal':
        SSD_model = SSLResNet(arch='resnet50').eval()
        if device =='cuda':
            SSD_model.encoder = nn.DataParallel(SSD_model.encoder).to(device)
        else:
            SSD_model.encoder = SSD_model.encoder.to(device)
        ckpt_dict = torch.load(os.path.join(cwd,'SSD','train_IMAGENET','latest_exp','checkpoint','model_best.pth.tar'), map_location="cpu")#checkpoint_3.pth.tar
    if "model" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["model"]
    if "state_dict" in ckpt_dict.keys():
        ckpt_dict = ckpt_dict["state_dict"]
    SSD_model.load_state_dict(ckpt_dict, strict=False)
    
    #SSD_model.to(device)
    
    #if opt.dataset == 'IMAGENET':
    #    features_train, labels_train = get_features(SSD_model.encoder, train_loader,device,max_images=100000) 
    #else:
    features_train, labels_train = get_features(SSD_model.encoder, train_loader,device,max_images=10000) 
    #features_train, labels_train = get_features(SSD_model.encoder, train_loader,device,max_images=10000)  # using feature befor MLP-head
    #print(features_train, labels_train)
    
    return SSD_model,features_train,  labels_train

def fake_train(opt, dataset='CIFAR10'):
    
    best_acc = 0
    M = create_model(opt)
    M.setup(opt)
    M.eval()
    def test(net, epoch,best_acc):
        
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(data.test):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = fast_gradient_method(net, inputs, eps=opt.eps,norm= opt.attack_norm)
            with torch.no_grad():
                inputs = M.netG(inputs)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(data.test), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            #if not os.path.isdir('checkpoint'):
            #    os.mkdir('checkpoint')
            torch.save(state, os.path.join(cwd,'CIFAR10','pytorch','checkpoint','DLA_fake.pth'))
            best_acc = acc
            
        return best_acc
            
    data = ld_cifar10(opt.dataroot,opt.batch_size,forGAN=False)
    #clf = SimpleDLA()
    clf = DLA()
    clf = clf.to(device)
    clf = torch.nn.DataParallel(clf)
    cudnn.benchmark = True
    
    # load checkpoint
    print('loading checkpoint')
    checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','DLA_fake.pth'))
    best_acc = checkpoint['acc']
    print('best_acc',best_acc)
    clf.load_state_dict(checkpoint['net'])
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    
    for epoch in range(0, 200):
        print('\nEpoch: %d' % epoch)
        clf.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(data.train):
            inputs, targets = inputs.to(device), targets.to(device)
            #inputs = fast_gradient_method(clf, inputs, eps=opt.eps,norm= opt.attack_norm)
            with torch.no_grad():
                inputs = M.netG(inputs)
            optimizer.zero_grad()
            outputs = clf(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            progress_bar(batch_idx, len(data.train), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        best_acc = test(clf,epoch,best_acc)
        scheduler.step()
    
    
    
    
        
        
def save_image_wrapper(img, path):
    
    img=tensor2im(img)
    #img=img.cpu()
    #img = img[0]
    
    #img=img.detach()
    save_image(img, path)
    
def main():
        
    
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
        
        
    # load data
    if opt.dataset == 'MNIST':
        nb_classes = 10
        data = ld_mnist(opt.dataroot,batch_size=opt.batch_size)
        # load pretrained classifier
        clf = PyNet()
        clf.to(device)
        clf.load_state_dict(torch.load(os.path.join(cwd,opt.dataset,'CNN_'+opt.dataset+'.pth')))
        clf.eval()
    elif opt.dataset == 'CIFAR10':
        nb_classes = 10
        print('Loading data')
        data = ld_cifar10(opt.dataroot,opt.batch_size,forGAN=False)
        print('Loading model')
        clf = SimpleDLA()
        #clf = DLA()
        clf = clf.to(device)
        
        #if device == 'cuda':
        clf = torch.nn.DataParallel(clf)
        cudnn.benchmark = True
        
        checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','DLA.pth'))
        clf.load_state_dict(checkpoint['net'])
        #checkpoint = torch.load(,map_location="cpu")#'ckpt.pth'))#
        #clf.load_state_dict(checkpoint['net'],strict=False)
        clf.eval()
    elif opt.dataset in ['sharp2normal', 'IMAGENET']:
        nb_classes = 100
        # load model
        clf = models.resnext50_32x4d(pretrained=True)
        clf.to(device)
        clf.eval()
        # load data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        if opt.dataset == 'IMAGENET' and opt.defense=='Im2Im':
            transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),normalize])
        elif opt.dataset == 'IMAGENET' and opt.defense=='adv_train':
            transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])#,normalize])
        else:
            transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])#,normalize])
            transform_SSD = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),normalize])
        imagenet_data_test = torchvision.datasets.ImageNet(os.path.join(opt.dataroot,'val'), split='val',transform=transform)
        imagenet_data_train = torchvision.datasets.ImageNet(os.path.join(opt.dataroot,'train'), split='train',transform=transform)
        data = EasyDict(train=torch.utils.data.DataLoader(imagenet_data_train,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers),
                        test=torch.utils.data.DataLoader(imagenet_data_test,
                                                  batch_size=opt.batch_size,
                                                  shuffle=False,
                                                  num_workers=opt.num_workers))
        if opt.dataset != 'IMAGENET':
            data_SSD = EasyDict(train=torch.utils.data.DataLoader(torchvision.datasets.ImageNet(os.path.join(opt.dataroot,'train'), split='train',transform=transform_SSD),
                                                      batch_size=opt.batch_size,
                                                      shuffle=False,
                                                      num_workers=opt.num_workers),
                            test=torch.utils.data.DataLoader(torchvision.datasets.ImageNet(os.path.join(opt.dataroot,'val'), split='val',transform=transform_SSD),
                                                      batch_size=opt.batch_size,
                                                      shuffle=False,
                                                      num_workers=opt.num_workers))
    elif opt.dataset == 'dark2clear':
       nb_classes = 10
       print('Loading data')
       data = ld_cifar10(opt.dataroot,opt.batch_size,forGAN=True)
       print('Loading model')
       clf = SimpleDLA()
       #clf = DLA()
       clf = clf.to(device)
       
       #if device == 'cuda':
       clf = torch.nn.DataParallel(clf)
       cudnn.benchmark = True
       
       checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','ckpt.pth')) # model trained in non-normalized data
       clf.load_state_dict(checkpoint['net'])
       #checkpoint = torch.load(,map_location="cpu")#'ckpt.pth'))#
       #clf.load_state_dict(checkpoint['net'],strict=False)
       clf.eval()
    
    ''''''
    if opt.defense == 'Im2Im':
        print('##### Setting up distribution filter ...')
        if opt.dataset != 'sharp2normal':
            SSD_model, features_train , labels_train= init_SSD(data.train, data.test,opt)
        else:
            SSD_model, features_train , labels_train= init_SSD(data_SSD.train, data_SSD.test,opt)
    
    if opt.defense == 'Im2Im':
        print('##### Setting up Im2Im model ...')
        if opt.name != 'pix_ens_M':
            M = create_model(opt)
            M.setup(opt)
            M.eval()
        else:
            if opt.dataset != 'CIFAR10':
                raise ValueError('{} is implemented only for CIFAR10'.format(opt.name))
            M={}
            M_names = ['pix_pgd2Cifar10_resnet','pix_spsa2cifar10','pix_fgs2cifar10']
            epochs = ['12','11','latest']
            ind = 0
            for M_name in M_names:
                opt.epoch = epochs[ind]
                opt.name = M_name
                M[M_name]=create_model(opt)
                M[M_name].setup(opt)
                M[M_name].eval()
                ind+=1
            opt.name = 'pix_ens_M'
    
    elif opt.defense == 'ens_adv':
        if opt.dataset == 'MNIST':
            ens_adv_model = PyNet()
            ens_adv_model.load_state_dict(torch.load(os.path.join(cwd,'ens_adv','MNIST','ensemble-adv-training-pytorch','models','CNN_MNIST_ens_adv.pkl')))
            ens_adv_model= ens_adv_model.to(device)
            ens_adv_model.eval()
        elif opt.dataset == 'CIFAR10':
            ens_adv_model = SimpleDLA()
            
            #if device == 'cuda':
            #ens_adv_model = torch.nn.DataParallel(ens_adv_model).to(device)
            #cudnn.benchmark = True
            
            checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','DLA_ens_adv_train.pth'))
            if 'state_dict' in checkpoint.keys():
                state = 'state_dict'
            elif 'net' in checkpoint.keys():
                state = 'net'
            ens_adv_model.load_state_dict(checkpoint[state])#,strict=False)
            ens_adv_model= ens_adv_model.to(device)
            ens_adv_model.eval()
        elif opt.dataset == 'IMAGENET':
            ens_adv_model = models.resnext50_32x4d(pretrained=False)
            checkpoint = torch.load(os.path.join(cwd,'ens_adv','CIFAR10','Ensemble-Adversarial-Training','ens_adv_imagenet.pth'))
            if 'state_dict' in checkpoint.keys():
                state = 'state_dict'
            elif 'net' in checkpoint.keys():
                state = 'net'
            ens_adv_model.load_state_dict(checkpoint[state],strict=False)
            ens_adv_model= ens_adv_model.to(device)
            ens_adv_model.eval()
    elif opt.defense == 'adv_train':
        if opt.dataset == 'MNIST':
            adv_model = PyNet()
            adv_model.load_state_dict(torch.load(os.path.join(cwd,'ens_adv','MNIST','ensemble-adv-training-pytorch','models','CNN_MNIST_adv.pkl')))
            adv_model= adv_model.to(device)
            adv_model.eval()
        elif opt.dataset == 'CIFAR10':
            adv_model  = SimpleDLA()
           
            #if device == 'cuda':
            adv_model = torch.nn.DataParallel(adv_model)
            cudnn.benchmark = True
           
            checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','DLA_adv_train.pth'))
            if 'state_dict' in checkpoint.keys():
                state = 'state_dict'
            elif 'net' in checkpoint.keys():
                state = 'net'
            adv_model.load_state_dict(checkpoint[state],strict=False)
            adv_model= adv_model.to(device)
            adv_model.eval()
        elif opt.dataset == 'IMAGENET':
            #adv_model = models.resnext50_32x4d()
            #adv_model = adv_model.load_from_checkpoint(os.path.join(cwd,'ens_adv','resnet50_l2_eps0.25.ckpt'))
            adv_model, checkpoint = model_utils.make_and_restore_model(arch='resnet50', dataset=rob_datasets.ImageNet(os.path.join(opt.dataroot,'train')), resume_path=os.path.join(cwd,'ens_adv','IMAGENET','resnet50_linf_eps2.0.ckpt'))
            
            adv_model= adv_model.to(device)
            adv_model.eval()
    else:
        raise ValueError('{} is not supported try [adv_train, ens_adv, or Im2Im]'.format(opt.defense))
    
    
    
    
        
    report = EasyDict(nb_test=0, correct=0, correct_adv=0, correct_M=0)
    
    i=0
    print('##### Accepting queries ...')
    for x, y in data.test:
        
        #print(y)
        #torch.Tensor([y]).to(device) #
        i+=opt.batch_size
        if opt.dataset in  ['IMAGENET','sharp2normal']:
            if i>5000:
                break
            mask = y<nb_classes
            y= y[mask]
            x= x[mask]
            #print(x.shape[0])
            if x.shape[0] == 0:
                continue
       
        x, y = x.to(device),  y.to(device)
        report.nb_test += y.size(0)
        if opt.dataset == 'sharp2normal':
            _, y_pred = clf(normalize(x)).max(1)
        else:
            if opt.dataset == 'IMAGENET' and opt.defense=='adv_train':
                _, y_pred = clf(normalize(x)).max(1)
            else:
                _, y_pred = clf(x).max(1)
        report.correct += y_pred.eq(y).sum().item()
        
        ''''''
        if opt.defense == 'Im2Im':
            if opt.dataset != 'sharp2normal':
                features_test, _ = get_features_1batch(SSD_model.encoder, x,y,attack=None)
            else:
                features_test, _ = get_features_1batch(SSD_model.encoder, normalize(x),y,attack=None)
        
        if opt.defense == 'Im2Im':
            if opt.dataset not in ['sharp2normal','dark2clear','IMAGENET2CIFAR10']:
                if opt.attack=='FGS':
                    x = fast_gradient_method(clf, x, eps=opt.eps,norm= opt.attack_norm)
                if opt.attack=='PGD':
                    if opt.eps>0.01:
                        x = projected_gradient_descent(clf, x, eps = opt.eps, nb_iter=40,  norm = opt.attack_norm, eps_iter=0.01) # 0.01, 40,
                    else:
                        x = projected_gradient_descent(clf, x, eps = opt.eps, nb_iter=40,  norm = opt.attack_norm, eps_iter=opt.eps-0.0001)
                if opt.attack=='CW':
                    if opt.dataset != 'IMAGENET':
                        x = carlini_wagner_l2(clf, x, nb_classes,y,targeted = False)
                    else:
                        x = carlini_wagner_l2(clf, x, 1000,y,targeted = False)
                if opt.attack=='SPSA':
                    x = spsa(clf, x,eps=opt.eps,nb_iter=500,norm = opt.attack_norm)#,sanity_checks=False)
            elif opt.dataset == 'dark2clear':
                # convert cifar10 into night model
                x=transforms.functional.adjust_contrast(x,contrast_factor=3.8)
                x=transforms.functional.adjust_brightness(x,brightness_factor=0.1)
            elif opt.dataset == 'sharp2normal':
                #x = transforms.functional.adjust_saturation(x,saturation_factor=6)
                x =transforms.functional.adjust_sharpness(x,sharpness_factor=15)
                
                
        elif opt.defense == 'ens_adv':
            if opt.attack=='FGS':
                x = fast_gradient_method(ens_adv_model, x, eps=opt.eps,norm= opt.attack_norm)
            if opt.attack=='PGD':
                x = projected_gradient_descent(ens_adv_model, x, eps = opt.eps, nb_iter=40,  norm = opt.attack_norm, eps_iter=0.01) # 0.01, 40,
            if opt.attack=='CW':
                x = carlini_wagner_l2(ens_adv_model, x, nb_classes,y,targeted = False)
            if opt.attack=='SPSA':
                x = spsa(ens_adv_model, x,eps=opt.eps,nb_iter=500,norm = opt.attack_norm)#,sanity_checks=False)
        elif opt.defense == 'adv_train':
            if opt.attack=='FGS':
                x = fast_gradient_method(adv_model, x, eps=opt.eps,norm= opt.attack_norm)
            if opt.attack=='PGD':
                if opt.eps>0.01:
                    x = projected_gradient_descent(adv_model, x, eps = opt.eps, nb_iter=40,  norm = opt.attack_norm, eps_iter=0.01) # 0.01, 40,
                else:
                    x = projected_gradient_descent(adv_model, x, eps = opt.eps, nb_iter=40,  norm = opt.attack_norm, eps_iter=opt.eps-0.0001) 
            if opt.attack=='CW':
                x = carlini_wagner_l2(adv_model, x, nb_classes,y,targeted = False)
            if opt.attack=='SPSA':
                x = spsa(adv_model, x,eps=opt.eps,nb_iter=500,norm = opt.attack_norm)#,sanity_checks=False)
            
        if opt.attack=='NoAttack':
            pass
        
        
        
        
        
        
        if opt.defense == 'Im2Im':
            
            ''''''
            if opt.dataset != 'sharp2normal':
                features_ood, _ = get_features_1batch(SSD_model.encoder, x,y,attack=None)
            else:
                features_ood, _ = get_features_1batch(SSD_model.encoder, normalize(x),y,attack=None)
            #print(features_train)
            SSD_score = get_eval_results(
                np.copy(features_train),
                np.copy(features_test),
                np.copy(features_ood),
                np.copy(labels_train),
                'SimCLR', 
                1,
            )
            print('Current OOD score ', SSD_score)
            
            if SSD_score>=10:
                print('Using M for OOD input')
                if opt.name != 'pix_ens_M':
                    print('Using M for OOD input')
                    with torch.no_grad():
                        x_tilde = M.netG(x)
                        if opt.dataset in  ['IMAGENET','sharp2normal']:
                            x_tilde=normalize(x_tilde)
                else:
                    print('Using enseble-GAN defense for OOD input')
                    x_tilde = {}
                    with torch.no_grad():
                        for M_name in M_names:
                            x_tilde[M_name]=M[M_name].netG(x)
            else:
                print('Input is in-distribution')
                x_tilde = x.clone()
                if opt.dataset == 'sharp2normal':
                    x_tilde=normalize(x_tilde)
            '''
            if opt.attack != 'NoAttack':
                if opt.name != 'pix_ens_M':
                    print('Using M for OOD input')
                    with torch.no_grad():
                        x_tilde = M.netG(x)
                        if opt.dataset in ['IMAGENET','sharp2normal']:
                            x_tilde=normalize(x_tilde)
                else:
                    print('Using enseble-GAN defense for OOD input')
                    x_tilde = {}
                    with torch.no_grad():
                        for M_name in M_names:
                            x_tilde[M_name]=M[M_name].netG(x)
                            
            else:
                print('Input is in-distribution')
                x_tilde = x.clone()
                if opt.dataset == 'sharp2normal':
                    x_tilde=normalize(x_tilde)
           ''' 
                #print(x_tilde.shape)
            
            if opt.name != 'pix_ens_M':
                _, y_pred = clf(x_tilde).max(1)
                report.correct_M += y_pred.eq(y).sum().item()
            else:
                
                y_preds = []
                ind = 0
                for M_name in M_names:
                    #print(clf(x_tilde[M_name]))
                    y_preds.append(clf(x_tilde[M_name]).max(1)[1].tolist())
                    if ind == 0:
                        y_probs = clf(x_tilde[M_name])
                    else:
                        y_probs+=clf(x_tilde[M_name])
                    ind+=1
                    
                # majority vote
                
                '''Uncomment the following to run ens-M with majority vote'''
                # y_preds = np.array(y_preds)
                # y_pred=[]
                # for ind in range(y_preds.shape[1]):
                #     y_pred.append(max(set(list(y_preds[:,ind])), key = list(y_preds[:,ind]).count))
                
                
                # y_pred = torch.Tensor(y_pred).to(device)
                
                # Highest confidence
                '''Comment the following line to run ens-M with majority vote'''
                _,y_pred = y_probs.max(1)
                
                
                
                
                report.correct_M += y_pred.eq(y).sum().item()
                
                    
        
        elif opt.defense == 'ens_adv':
            _, y_pred = ens_adv_model(x).max(1)
            report.correct_M += y_pred.eq(y).sum().item()
        elif opt.defense == 'adv_train':
            #print(adv_model(x)[0].shape)
            #print(adv_model(x)[1].shape)
            _, y_pred = adv_model(x).max(1)
            report.correct_M += y_pred.eq(y).sum().item()
        #save_image_wrapper(x_tilde,os.path.join(cwd,opt.results_dir,opt.name,'test_latest','fakeimg{}_{}_({}).png'.format(report.nb_test,y.item(),y_pred.item())))
        
        
        # Accuracy without M
        if opt.dataset == 'sharp2normal':
            _, y_pred = clf(normalize(x)).max(1)
        else:
            _, y_pred = clf(x).max(1)
        report.correct_adv += y_pred.eq(y).sum().item()
        
        
        
        #if report.nb_test % 50 == 0:
        if opt.dataset not in ['sharp2normal','dark2clear','IMAGENET2CIFAR10']:
            print('Current accuracy on {} samples without attack is {}'.format(report.nb_test,report.correct / report.nb_test * 100.0))
            
            if opt.defense == 'Im2Im':
                print('Current accuracy on {} samples Under {} without {} is {}'.format(report.nb_test,opt.attack,opt.defense,report.correct_adv / report.nb_test * 100.0))
            print('Current accuracy on {} samples Under {} after using {} is {}'.format(report.nb_test, opt.attack, opt.defense,report.correct_M / report.nb_test * 100.0))
            
            # Computing relative robustness
            if report.correct!=0:
                if opt.defense == 'Im2Im':
                    print('Current RR on {} samples Under {} without {} is {}'.format(report.nb_test,opt.attack,opt.defense,report.correct_adv/report.correct * 100.0))
                print('Current RR on {} samples Under {} after using {} is {}\n'.format(report.nb_test,opt.attack, opt.defense,report.correct_M / report.correct * 100.0))
            else:
                if opt.defense == 'Im2Im':
                    print('Current RR on {} samples Under {} without {} is {}'.format(report.nb_test,opt.attack, opt.defense, report.correct_adv/report.nb_test * 100.0))
                print('Current RR on {} samples Under {} after using {} is {}\n'.format(report.nb_test,opt.attack, opt.defense, report.correct_M / report.nb_test * 100.0))
        else:
             if opt.dataset == 'dark2clear':
                 print('Current accuracy of CIFAR10 model on  {} day light samples is {}'.format(report.nb_test,report.correct / report.nb_test * 100.0))
                 print('Current accuracy of CIFAR10 model on  {} dark mode samples before using {} is {}'.format(report.nb_test, opt.defense,report.correct_adv / report.nb_test * 100.0))
                 print('Current accuracy of CIFAR10 model on  {} dark mode samples after using {} is {}\n'.format(report.nb_test, opt.defense,report.correct_M / report.nb_test * 100.0))
             elif opt.dataset == 'sharp2normal':
                 print('Current accuracy of IMAGENET model on  {} original samples is {}'.format(report.nb_test,report.correct / report.nb_test * 100.0))
                 print('Current accuracy of IMAGENET model on  {} sharper samples before using {} is {}'.format(report.nb_test, opt.defense,report.correct_adv / report.nb_test * 100.0))
                 print('Current accuracy of IMAGENET model on  {} sharper samples after using {} is {}\n'.format(report.nb_test, opt.defense,report.correct_M / report.nb_test * 100.0))
        #i+=opt.batch_size
        #if i>=10000:
        #    break
        #save_image_wrapper(x,os.path.join(cwd,opt.results_dir,opt.name,'test_latest','PGDimg{}_{}_({}).png'.format(report.nb_test,y.item(),y_pred.item())))
        #break
        



    
# local utils for SSD evaluation
def get_scores(ftrain, ftest, food, labelstrain, training_mode, clusters):
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        if training_mode == "SupCE":
            print("Using data labels as cluster since model is cross-entropy")
            ypred = labelstrain
        else:
            ypred = get_clusters(ftrain, clusters)
        return get_scores_multi_cluster(ftrain, ftest, food, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood


def get_eval_results(ftrain, ftest, food, labelstrain, training_mode, clusters):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food, labelstrain, training_mode, clusters)
    
    
    #print('dtest = {}'.format(dtest))
    #print('dood = {}'.format(dood))
    
    #diff=[]
    outofd=0
    dtout = {}
    dtin = {}
    print(len(dtest))
    for ind in range(len(dood)):
        #diff.append(dood[ind]-dtest[ind])
        dtout[ind]=dood[ind]
        dtin[ind]=dtest[ind]
        if dtest[ind]<dood[ind]:
            outofd+=1
    #print('{} % are out of distribution'.format(outofd/len(dtest)*100))
    #print(diff)
    #x = list(dtout.keys())
    #yout = list(dtout.values())
    #yin = list(dtin.values())
    #plot(x,yin,yout)
    
    
    
    #fpr95 = get_fpr(dtest, dood)
    #auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return outofd/len(dtest)*100 #fpr95, auroc, aupr


if __name__ == '__main__':
    
    main()
