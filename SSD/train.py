# Some part borrowed from official tutorial https://github.com/pytorch/examples/blob/master/imagenet/main.py
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import argparse
import importlib
import time
import logging
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from models import SupResNet, SSLResNet
import data
import trainers
from losses import SupConLoss
from utils import *


def main():
    parser = argparse.ArgumentParser(description="SSD evaluation")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="/data/data_vvikash/fall20/SSD/trained_models/",
    )  # change this
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument(
        "--training-mode", type=str, choices=("SimCLR", "SupCon", "SupCE")
    )

    # model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--in_channel", type=int, default=3)

    # training
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, default="/data/data_vvikash/datasets/")
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--continue", action="store_true")

    # ssl
    parser.add_argument(
        "--method", type=str, default="SupCon", choices=["SupCon", "SimCLR", "SupCE"]
    )
    parser.add_argument("--temperature", type=float, default=0.5)

    # misc
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--ckpt", default="", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=12345)

    args = parser.parse_args()
    device = "cuda:0"

    if args.batch_size > 256 and not args.warmup:
        warnings.warn("Use warmup training for larger batch-sizes > 256")

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # create resutls dir (for logs, checkpoints, etc.)
    result_main_dir = os.path.join(args.results_dir, args.exp_name)

    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = result_sub_dir = os.path.join(
            result_main_dir,
            "{}--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                n + 1, args.dataset, args.arch, args.lr, args.epochs
            ),
        )
    else:
        os.mkdir(result_main_dir)
        result_sub_dir = result_sub_dir = os.path.join(
            result_main_dir,
            "1--dataset-{}-arch-{}-lr-{}_epochs-{}".format(
                args.dataset, args.arch, args.lr, args.epochs
            ),
        )
    create_subdirs(result_sub_dir)

    # add logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "setup.log"), "a")
    )
    logger.info(args)

    # seed cuda
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create model
    if args.training_mode in ["SimCLR", "SupCon"]:
        model = SSLResNet(arch=args.arch, in_channel=args.in_channel).to(device)
    elif args.training_mode == "SupCE":
        model = SupResNet(arch=args.arch, num_classes=args.num_classes, in_channel=args.in_channel).to(device)
    else:
        raise ValueError("training mode not supported")
    
    # load checkpoint
    if args.ckpt != '':
        print('loading checkpoint...')
        if device =='cuda':
            model.encoder = nn.DataParallel(model.encoder).to(device)
        else:
            model.encoder = model.encoder.to(device)
        ckpt_dict = torch.load(args.ckpt, map_location="cpu")
        if "model" in ckpt_dict.keys():
            ckpt_dict = ckpt_dict["model"]
        if "state_dict" in ckpt_dict.keys():
            ckpt_dict = ckpt_dict["state_dict"]
        model.load_state_dict(ckpt_dict, strict=False)
    else:
        # load feature extractor on gpu
        model.encoder = torch.nn.DataParallel(model.encoder).to(device)

    # Dataloader
    '''
    if args.dataset == 'IMAGENET':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        
        transform = transforms.Compose([transforms.Resize(args.size), transforms.CenterCrop(args.size), transforms.ToTensor(),normalize])
        imagenet_data_train = torchvision.datasets.ImageNet(os.path.join(args.data_dir,'train'), split='train',transform=transform)
        train_loader = torch.utils.data.DataLoader(imagenet_data_train,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
        imagenet_data_test = torchvision.datasets.ImageNet(os.path.join(args.data_dir,'val'), split='val',transform=transform)
        test_loader = torch.utils.data.DataLoader(imagenet_data_test,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=2)
    else:
    '''
    train_loader, test_loader, _ = data.__dict__[args.dataset](
        args.data_dir,
        mode="ssl" if args.training_mode in ["SimCLR", "SupCon"] else "org",
        normalize=args.normalize,
        size=args.size,
        batch_size=args.batch_size,
    )
    
    criterion = (
        SupConLoss(temperature=args.temperature).cuda()
        if args.training_mode in ["SimCLR", "SupCon"]
        else nn.CrossEntropyLoss().cuda()
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # select training and validation methods
    trainer = (
        trainers.ssl
        if args.training_mode in ["SimCLR", "SupCon"]
        else trainers.supervised
    )
    val = knn if args.training_mode in ["SimCLR", "SupCon"] else baseeval

    # warmup
    if args.warmup:
        wamrup_epochs = 10
        print(f"Warmup training for {wamrup_epochs} epochs")
        warmup_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.01,
            max_lr=args.lr,
            step_size_up=wamrup_epochs * len(train_loader),
        )
        for epoch in range(wamrup_epochs):
            trainer(
                model,
                device,
                train_loader,
                criterion,
                optimizer,
                warmup_lr_scheduler,
                epoch,
                args,
            )

    best_prec1 = 0

    for p in optimizer.param_groups:
        p["lr"] = args.lr
        p["initial_lr"] = args.lr
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), 1e-4
    )

    for epoch in range(0, args.epochs):
        trainer(
            model, device, train_loader, criterion, optimizer, lr_scheduler, epoch, args
        )
        
        if args.training_mode in ["SimCLR", "SupCon"]:
            prec1, _ = val(model, device, test_loader,args)
        else:
            prec1, _ = val(model, device, test_loader, criterion, args, epoch)

        # remember best accuracy and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Current best_prec',best_prec1)
        d = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            #"optimizer": optimizer.state_dict(),
        }

        save_checkpoint(
            d,
            is_best,
            os.path.join(result_sub_dir, "checkpoint"),
        )

        if not (epoch + 1) % args.save_freq:
            save_checkpoint(
                d,
                is_best,
                os.path.join(result_sub_dir, "checkpoint"),
                filename=f"checkpoint_{epoch+1}.pth.tar",
            )

        logger.info(
            f"Epoch {epoch}, validation accuracy {prec1}, best_prec {best_prec1}"
        )

        # clone results to latest subdir (sync after every epoch)
        clone_results_to_latest_subdir(
            result_sub_dir, os.path.join(result_main_dir, "latest_exp")
        )


if __name__ == "__main__":
    main()
