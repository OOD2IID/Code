"""General-purpose training script for image-to-image translation.
This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from train_mnist import PyNet, ld_mnist
from CIFAR10.pytorch.train_cifar10 import ld_cifar10
from CIFAR10.pytorch.models import *
import numpy as np
from easydict import EasyDict

import torch
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.models as models
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
# Get current working directory
cwd = os.getcwd()

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    # create execution device
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
    #print(device)
    
    #if torch.cuda.is_available() and opt.dataset == 'CIFAR10':
    #    device = 'cuda'
     
    # load data
    
    if opt.dataset == 'MNIST':
        dataset = ld_mnist(opt.dataroot,opt.batch_size)
        dataset_size = 500
        nb_classes = 10
        clf = PyNet()
        clf.to(device)
        clf.load_state_dict(torch.load(os.path.join(cwd,'MNIST','CNN_MNIST.pth')))
        clf.eval()
    elif opt.dataset == 'CIFAR10':
        dataset = ld_cifar10(opt.dataroot,opt.batch_size,forGAN=False)
        dataset_size = 1000
        nb_classes = 10
        clf = SimpleDLA()
        clf = clf.to(device)
        
        if device != 'cpu':
            clf = torch.nn.DataParallel(clf)
            cudnn.benchmark = True
        
        checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','DLA.pth'))#ckpt.pth))
        clf.load_state_dict(checkpoint['net'])#,strict=False)
        clf.eval()
    elif opt.dataset == 'dark2clear':
        dataset = ld_cifar10(opt.dataroot,opt.batch_size,forGAN=True)
        dataset_size = 1000
        nb_classes = 10
        clf = SimpleDLA()
        clf = clf.to(device)
        
        if device != 'cpu':
            clf = torch.nn.DataParallel(clf)
            cudnn.benchmark = True
        
        checkpoint = torch.load(os.path.join(cwd,'CIFAR10','pytorch','checkpoint','ckpt.pth'))#ckpt.pth))
        clf.load_state_dict(checkpoint['net'])#,strict=False)
        clf.eval()
    elif opt.dataset in  ['IMAGENET','sharp2normal']:
        dataset_size = 1000
        nb_classes = 100
        # load model
        clf = models.resnext50_32x4d(pretrained=True)
        clf.to(device)
        clf.eval()
        # load data
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])#,normalize])
        imagenet_data = torchvision.datasets.ImageNet(os.path.join(opt.dataroot), split='train',transform=transform)
        dataset = EasyDict(train=torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True,
                                                  num_workers=opt.num_workers),test='')
    elif opt.dataset == 'IMAGENET2CIFAR10':
        class_mapping = {726:0,
                         895:0,
                         436:1,
                         751:1,
                         817:1,
                         829:1,
                         13:2,
                         14:2,
                         94:2,
                         281:3,
                         282:3,
                         283:3,
                         284:3,
                         285:3,
                         383:3,
                         177:5,153:5,200:5,229:5,230:5,235:5,238:5,239:5,245:5,248:5,251:5,252:5,254:5,256:5,275:5,
                         30:6,31:6,32:6,
                         510:8,724:8,554:8,625:8,814:8,
                         864:9,555:9,569:9,717:9,864:9,867:9}
        opt.dataroot = opt.dataroot.split(',')
        #print(opt.dataroot)
        if len(opt.dataroot) != 2:
            raise ValueError('For IMAGENET2CIFAR10 experiment --dataroot should be equal to a list [IMAGENET dataroot,CIFAR10 dataroot]')
        else:
            cifar10_dataset = ld_cifar10(opt.dataroot[1],opt.batch_size,forGAN=True)
            dataset_size = 2000
            nb_classes = 10
            transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor()])#,normalize])
            imagenet_data = torchvision.datasets.ImageNet(os.path.join(opt.dataroot[0]), split='train',transform=transform)
            imagenet_dataset = EasyDict(train=torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=opt.batch_size,
                                                  shuffle=True,
                                                  num_workers=opt.num_workers),test='')
            
        
        #if device == 'cuda':
        #    clf = torch.nn.DataParallel(clf)
        #    cudnn.benchmark = True
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        #for i, data in enumerate(dataset):  # inner loop within one epoch
        i=0
        if opt.dataset != 'IMAGENET2CIFAR10':
            for x, y in dataset.train:
                #print(y.item())
                #print(x.shape)
                if opt.batch_size == 1:
                    if opt.dataset in ['IMAGENET','sharp2normal'] and y.item() > 100:
                        continue
                
                
                if opt.batch_size*i>dataset_size:
                    break
                
                x=x.to(device)
                y=y.to(device)
                ## comment if normalized data is used
                
                if opt.dataset == 'IMAGENET':
                    x_adv = x.clone()
                    x_adv = x_adv.to(device)
                    x_adv = normalize(x_adv)
                    if opt.attack=='FGS':
                        x_adv = fast_gradient_method(clf,x_adv , eps=opt.eps,norm= opt.attack_norm)
                    elif opt.attack=='PGD':
                        x_adv = projected_gradient_descent(clf, x_adv, opt.eps, 0.01, 40, opt.attack_norm)
                    elif opt.attack=='CW':
                        x_adv = carlini_wagner_l2(clf, x_adv, nb_classes,y,targeted = False)
                    elif opt.attack=='SPSA':
                        x_adv = spsa(clf, x_adv,eps=opt.eps,nb_iter=1000,norm = opt.attack_norm)#,sanity_checks=False)
                    elif opt.attack=='MIX':
                        if opt.batch_size*i<dataset_size/3:
                            x_adv = fast_gradient_method(clf,x_adv , eps=opt.eps,norm= opt.attack_norm)
                        elif opt.batch_size*i>=dataset_size/3 and opt.batch_size*i<2*(dataset_size/3):
                            x_adv = projected_gradient_descent(clf, x_adv, opt.eps, 0.01, 40, opt.attack_norm)
                        else:
                            x_adv = carlini_wagner_l2(clf, x_adv, nb_classes,y,targeted = False)
                    else:
                        raise ValueError('{} is not a supported, please select from [FGS,PGD,CW,SPSA and MIX]'.format(opt.attack))
                  
                elif opt.dataset not in ['dark2clear','sharp2normal']:
                    if opt.attack=='FGS':
                        x_adv = fast_gradient_method(clf, x, eps=opt.eps,norm= opt.attack_norm)
                    elif opt.attack=='PGD':
                        x_adv = projected_gradient_descent(clf, x, opt.eps, 0.01, 40, opt.attack_norm) # 0.01, 40,
                    elif opt.attack=='CW':
                        x_adv = carlini_wagner_l2(clf, x, nb_classes,y,targeted = False)
                    elif opt.attack=='SPSA':
                        x_adv = spsa(clf, x,eps=opt.eps,nb_iter=1000,norm = opt.attack_norm)#,sanity_checks=False)
                    elif opt.attack=='MIX':
                        if opt.batch_size*i<dataset_size/3:
                            x_adv = fast_gradient_method(clf,x , eps=opt.eps,norm= opt.attack_norm)
                        elif opt.batch_size*i>=dataset_size/3 and opt.batch_size*i<2*(dataset_size/3):
                            x_adv = projected_gradient_descent(clf, x, opt.eps, 0.01, 40, opt.attack_norm)
                        else:
                            x_adv = carlini_wagner_l2(clf, x, nb_classes,y,targeted = False)
                    else:
                        raise ValueError('{} is not a supported, please select from [FGS,PGD,CW,SPSA and MIX]'.format(opt.attack))
                elif opt.dataset == 'dark2clear':
                    x_adv=transforms.functional.adjust_contrast(x,contrast_factor=3.8)
                    x_adv = transforms.functional.adjust_brightness(x_adv,brightness_factor=0.1)
                elif opt.dataset == 'sharp2normal':
                    x_adv =transforms.functional.adjust_sharpness(x,sharpness_factor=15)
                    #x_adv = transforms.functional.adjust_saturation(x,saturation_factor=6)
                    
                    
                data = {}
                
                #print('x_pgd',x_pgd.shape)
                #if opt.model == 'pix2pix':
                with torch.no_grad():
                    #if opt.attack=='PGD':
                    data['A'] = x_adv.clone()
                    data['B']=x
                    data['A_paths'] = ''
                    data['B_paths'] = ''
                #print(data)
                #print(data['A'].shape)
                i+=1
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
    
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                
                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
    
                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
    
                iter_data_time = time.time()
        else:
            #for cifar10_batch, imagenet_batch in zip(cifar10_dataset.train, imagenet_dataset.train):
            for imagenet_batch in imagenet_dataset.train:
            #print(y.item())
            #print(x.shape)
                if opt.batch_size == 1:
                    
                    # filter imagenet data into only classes that are supported by cifar10 classes = ('plane':[726,895], 'car':[436,751,817,829], 'bird':[13,14,94],
                    #'cat':[281,282,283,284,285,383], 'deer':NA,'dog':[177,153,200,229,230,235,238,239,245,248,251,252,254,256,275], 'frog':[30,31,32], 'horse':NA, 'ship':[510,724,554,625,814]
                    #, 'truck':[864,555,569,717,864,867])
                    if imagenet_batch[1].item() not in [726,895,436,751,817,829,864,13,14,94,281,282,283,284,285,383,177,153,200,229,230,235,238,239,245,248,251,252,
                                        254,256,275,30,31,32,510,724,554,625,814,555,569,717,864,867]:
                        continue
                    
                       
                
                
                if opt.batch_size*i>dataset_size:
                    break
                
                for cifar10_batch in cifar10_dataset.train:
                    y=class_mapping[imagenet_batch[1].item()]
                    if cifar10_batch[1].item() != y:
                        continue
                    else:
                        break
                
                x_source = imagenet_batch[0].to(device)
                x_target = cifar10_batch[0].to(device)
                
                
                
                
                data = {}
                
                #print('x_pgd',x_pgd.shape)
                #if opt.model == 'pix2pix':
                with torch.no_grad():
                    #if opt.attack=='PGD':
                    data['A'] = x_source
                    data['B']=x_target
                    data['A_paths'] = ''
                    data['B_paths'] = ''
                #print(data)
                #print(data['A'].shape)
                i+=1
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
    
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
                
                if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                
                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
    
                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
    
                iter_data_time = time.time()
            
            
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
