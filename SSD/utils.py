import os
import sys
import numpy as np
import math
import time
import shutil, errno
from easydict import EasyDict
from distutils.dir_util import copy_tree
import sklearn.metrics as skm
from sklearn.covariance import ledoit_wolf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
# Get current working directory
cwd = os.getcwd()
#### logging ####
def save_checkpoint(state, is_best, results_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(results_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(results_dir, filename),
            os.path.join(results_dir, "model_best.pth.tar"),
        )


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


#### evaluation ####
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def get_features_1batch(model, img, label, verbose=False,attack=None):
    features, labels = [], []
    
    
    model.eval()
  

    img, label = img.cuda(), label.cuda()
    if attack!=None:
        clf = CNN()
        clf.cuda()
        clf.load_state_dict(torch.load(os.path.join(cwd,'CNN_CIFAR10.pth')))
        if attack=='FGS':
            img=fast_gradient_method(clf, img, eps=0.3, norm = np.inf)
        if attack=='PGD':
            img = projected_gradient_descent(clf, img, 0.3, 0.01, 40, np.inf)
        clf.eval()
        _, y_pred = clf(img).max(1)
        
        
    features += list(model(img).data.cpu().numpy())
    labels += list(label.data.cpu().numpy())

    return np.array(features), np.array(labels)
    
def get_features(model, dataloader,device, max_images=10 ** 10, verbose=False,attack=None):
    features, labels = [], []
    total = 0
    
    model.eval()
    report = EasyDict(nb_test=0, correct=0)
    for img, label in dataloader:

        if total > max_images:
            break

        img, label = img.to(device), label.to(device)
        if attack!=None:
            clf = CNN()
            clf.cuda()
            clf.load_state_dict(torch.load(os.path.join(cwd,'CNN_CIFAR10.pth')))
            if attack=='FGS':
                img=fast_gradient_method(clf, img, eps=0.3, norm = np.inf)
            if attack=='PGD':
                img = projected_gradient_descent(clf, img, 0.3, 0.01, 40, np.inf)
            clf.eval()
            _, y_pred = clf(img).max(1)
            report.nb_test += label.size(0)
            report.correct += y_pred.eq(label).sum().item()
            print('Current accuracy Under {} is {}'.format(attack,report.correct / report.nb_test * 100.0))
            
        features += list(model(img).data.cpu().numpy())
        labels += list(label.data.cpu().numpy())

        if verbose and not index % 50:
            print(index)

        total += len(img)

    return np.array(features), np.array(labels)


def baseeval(model, device, val_loader, criterion, args, epoch=0):
    """
    Evaluating on validation set inputs.
    """
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

        progress.display(i)  # print final results

    return top1.avg, top5.avg


def knn(model, device, val_loader,args):#, criterion, args, writer, epoch=0):
    """
    Evaluating knn accuracy in feature space.
    Calculates only top-1 accuracy (returns 0 for top-5)
    """

    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            
            images, target = data[0].cuda(), data[1]
            
            # compute output
            output = F.normalize(model(images), dim=-1).data.cpu()
            features.append(output)
            labels.append(target)
            if args.dataset == 'IMAGENET' and i>=5000:
                break
        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        cls = KNeighborsClassifier(20, metric="cosine").fit(features, labels)
        acc = 100 * np.mean(cross_val_score(cls, features, labels))

        print(f"knn accuracy for test data = {acc}")

    return acc, 0


#### OOD detection ####
def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    #print(labels)
    #print(data)
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_pr_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr = skm.average_precision_score(labels, data)
    return aupr


def get_fpr(xin, xood):
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)


def get_scores_one_cluster(ftrain, ftest, food, shrunkcov=False):
    if shrunkcov:
        print("Using ledoit-wolf covariance estimator.")
        cov = lambda x: ledoit_wolf(x)[0]
    else:
        cov = lambda x: np.cov(x.T, bias=True)

    # ToDO: Simplify these equations
    dtest = np.sum(
        (ftest - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (ftest - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    dood = np.sum(
        (food - np.mean(ftrain, axis=0, keepdims=True))
        * (
            np.linalg.pinv(cov(ftrain)).dot(
                (food - np.mean(ftrain, axis=0, keepdims=True)).T
            )
        ).T,
        axis=-1,
    )

    return dtest, dood


#### Dataloaders ####
def readloader(dataloader):
    images = []
    labels = []
    for img, label in dataloader:
        images.append(img)
        labels.append(label)
    return torch.cat(images), torch.cat(labels)


def unnormalize(x, norm_layer):
    m, s = (
        torch.tensor(norm_layer.mean).view(1, 3, 1, 1),
        torch.tensor(norm_layer.std).view(1, 3, 1, 1),
    )
    return x * s + m


class ssdk_dataset(torch.utils.data.Dataset):
    def __init__(self, images, norm_layer, copies=1, s=32):
        self.images = images

        # immitating transformations used at training self-supervised models 
        # replace it if training models with a different data augmentation pipeline
        self.tr = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(s, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                norm_layer,
            ]
        )

        self.n = len(images)
        self.size = len(images) * copies

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.tr(self.images[idx % self.n]), 0


def sliceloader(dataloader, norm_layer, k=1, copies=1, batch_size=128, size=32):

    images, labels = readloader(dataloader)
    indices = np.random.permutation(np.arange(len(images)))
    images, labels = images[indices], labels[indices]

    index_k = torch.cat(
        [torch.where(labels == i)[0][0:k] for i in torch.unique(labels)]
    ).numpy()
    index_not_k = np.setdiff1d(np.arange(len(images)), index_k)

    dataset_k = ssdk_dataset(
        unnormalize(images[index_k], norm_layer), norm_layer, copies, size
    )
    dataset_not_k = torch.utils.data.TensorDataset(
        images[index_not_k], labels[index_not_k]
    )
    print(
        f"Number of selected OOD images (k * num_classes_ood_dataset) = {len(index_k)} \nNumber of OOD images after augmentation  = {len(dataset_k)} \nRemaining number of test images in OOD dataset = {len(dataset_not_k)}"
    )

    loader_k = torch.utils.data.DataLoader(
        dataset_k, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    loader_not_k = torch.utils.data.DataLoader(
        dataset_not_k,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return loader_k, loader_not_k





class CNN(nn.Module):
    """Sample model."""

    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)

        return x