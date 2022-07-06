# Code
Pytorch implementation related to the submission : Towards Machine Learning Models Resilient to
Adversarial and Natural Distribution Shifts. Our code is implemented on top of [Pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## Requirements
- GPU environment with cuda memory
- Python 3.7.9
- Create a new python environment
- It is recommended to seperately install a stable torch version that is best compatible with your cuda and python versions through: https://pytorch.org/

## Installation

Run the following:

``` cd CycleGAN_pix2pix```

```pip install -r requirements.txt```

```cd ..```

## Datasets
Our code supports MNIST, CIFAR10 and ImageNet. If you want to run it on ImageNet you need to download the data manually from: https://image-net.org/ using the following steps:

- Signup and get access from ImageNet creators
- Go to ```Download```. In the ```ImageNet Large-scale Visual Recognition Challenge (ILSVRC)``` section click on ```2012```
- Click on ```Training images (Task 1 & 2)```, then click on ```Validation images (all tasks).```
- You need to create a folder called ```IMAGENET``` in ```./CycleGAN_pix2pix```, that contains two folders: ```train``` and ```val```
- Extract the training subsets in ```train``` and the validation subsets in ```val```.
- Lastly, download ```ILSVRC2012_devkit_t12.tar.gz``` and place it in ```./CycleGAN_pix2pix/IMAGENET/train``` and ```./CycleGAN_pix2pix/IMAGENET/val```

## Running experiments using pre-trained Im2Im translators

- Download our [pretrained Im2Im models](https://drive.google.com/file/d/1VzP9XWSEFrNFU2qhOgAo8QAoK-iJrWx4/view?usp=sharing) and store them in ```CycleGAN_pix2pix/checkpoints```
- With respect to the experiments that you are interested to reproduce, download the following pretrained models for OOD detection and adversrial training:

|Description|model | store in           |
|----------|:------------:|:--------------------------:|
|Ens-Adv-train on CIFAR10 |[DLA_ens_adv_train.pth](https://drive.google.com/file/d/1jiahk3lkV8bLW2ofEPS1l-jcChSWN35i/view?usp=sharing)| ./CIFAR10/pytorch/checkpoint|
|Adv-train on ImageNet |[resnet50_linf_eps2.0.ckpt](https://drive.google.com/file/d/1uy6YcqFEs58tlQE4WYSeR6rgfF4yNUf4/view?usp=sharing) [[1](https://github.com/Microsoft/robust-models-transfer)]  |./ens_adv/IMAGENET|
|Ens-Adv-train on ImageNet|[ens_adv_imagenet.pth](https://drive.google.com/file/d/1QBJyGE-hO8UohbvmFbjVIKDJFz2rBRzQ/view?usp=sharing)|/ens_adv/CIFAR10/Ensemble-Adversarial-Training|
|SSD model on CIFAR10|[CIFAR10.pth](https://drive.google.com/file/d/17WZP5ZWs0qwbnu0K96leas1R97b3RMbK/view?usp=sharing)|./SSD/|
|SSD model on ImageNet|[model_best.pth](https://drive.google.com/file/d/1tqxUjeUhAYdecN8GTmtufFZnWYiU8TiI/view?usp=sharing)|/SSD/train_IMAGENET/latest_exp/checkpoint|

Note: All other models are already included in the repo code.

### Testing our approach with a standalone M=pix2pix:

```python predict.py --dataroot CycleGAN_pix2pix/[data-folder] --checkpoints_dir CycleGAN_pix2pix/checkpoints --name [pix2pix-model-name] --model pix2pix  --results_dir CycleGAN_pix2pix/results --input_nc [number-of-input-channels] --output_nc [number-of-output-channels] --preprocess none --dataset [dataset-name] --batch_size [batch-size] --attack [Attack-name] --epoch [epoch-number-of-translator] --defense Im2Im --netG resnet_6blocks```

- CycleGAN_pix2pix/[data-folder] can be : CycleGAN_pix2pix/MNIST, CycleGAN_pix2pix/CIFAR10 or CycleGAN_pix2pix/IMAGENET
- "--name" should start with the prefix "pix_". Refer to the configuration table below.
- If you face cuda: out of memory error, try to reduce the [batch_size]
- [input_nc] = [output_nc] = 3 for CIFAR10 and ImageNet, while both are equal to 1 for MNIST.
- [dataset-name] can be MNIST, CIFAR10 or IMAGENET.
- To reproduce our results, refer to ```Best input configurations``` section below and use a maximum of eps equal to 0.3 for MNIST, 0.2 for CIFAR10 and 8/255~0.031 for IMAGENET.
- [Attack-name] can be either ```FGS```, ```PGD```,```SPSA```,```CW```or```NoAttack```.
- For faster experiments, start with ```FGS``` and ```PGD```. experiments on ```SPSA```and```CW``` takes a while to finish due to the high performance overhead of both attacks

Exemple: ```python predict.py --dataroot CycleGAN_pix2pix/MNIST --checkpoints_dir CycleGAN_pix2pix/checkpoints --name pix_pgd2mnist_resnet --model pix2pix --results_dir CycleGAN_pix2pix/results --input_nc 1 --output_nc 1 --preprocess none --dataset MNIST --batch_size 100 --attack FGS --epoch 37 --defense Im2Im --netG resnet_6blocks```

### Testing our approach with a standalone M=Cycle-GAN:

```python predict.py --dataroot CycleGAN_pix2pix/[data-folder] --checkpoints_dir CycleGAN_pix2pix/checkpoints --name [Cycle-GAN-model-name] --model test --no_dropout --results_dir CycleGAN_pix2pix/results --input_nc [number-of-input-channels] --output_nc [number-of-output-channels] --preprocess none --dataset [dataset-name] --batch_size [batch-size] --attack [Attack-name] --eps [attack-size] --epoch [epoch-number-of-translator] --defense Im2Im```

Exemple: ```python predict.py --dataroot CycleGAN_pix2pix/MNIST --checkpoints_dir CycleGAN_pix2pix/checkpoints --name pgd2mnist --model test --no_dropout --results_dir CycleGAN_pix2pix/results --input_nc 1 --output_nc 1 --preprocess none --dataset MNIST --batch_size 100 --attack FGS --eps 0.3 --epoch 7 --defense Im2Im```

### Testing our approach with an Ensemble Translators:

```python predict.py --dataroot CycleGAN_pix2pix/CIFAR10 --checkpoints_dir CycleGAN_pix2pix/checkpoints --name pix_ens_M --model pix2pix --results_dir CycleGAN_pix2pix/results --input_nc 3 --output_nc 3 --preprocess none --dataset CIFAR10 --batch_size 100 --attack FGS --defense Im2Im --netG resnet_6blocks --eps 0.2```

### Testing Adversarial Training:

```python predict.py --dataroot CycleGAN_pix2pix/[data-folder] --dataset [dataset-name] --batch_size [batch-size] --attack [Attack-name] --eps [attack-size] --defense Adv_train```

Exemple: ```python predict.py --dataroot CycleGAN_pix2pix/MNIST --dataset MNIST --batch_size 100 --attack FGS --eps 0.3 --defense adv_train```

### Testing Ensemble Adversarial Training:

```python predict.py --dataroot CycleGAN_pix2pix/[data-folder] --dataset [dataset-name] --batch_size [batch-size] --attack [Attack-name] --eps [attack-size] --defense ens_adv```

Exemple: ```python predict.py --dataroot CycleGAN_pix2pix/MNIST --dataset MNIST --batch_size 100 --attack FGS --eps 0.3 --defense ens_adv```

### Testing dark2clear:

```python predict.py --dataroot CycleGAN_pix2pix/CIFAR10 --checkpoints_dir CycleGAN_pix2pix/checkpoints --name pix_dark2clear2 --model pix2pix --results_dir CycleGAN_pix2pix/results --input_nc 3 --output_nc 3 --preprocess none --dataset dark2clear  --batch_size 100 --epoch latest  --netG resnet_6blocks```

### Testing sharp2normal:
```python predict.py --dataroot CycleGAN_pix2pix/IMAGENET --checkpoints_dir CycleGAN_pix2pix/checkpoints --name pix_sharp2normal --model pix2pix --results_dir CycleGAN_pix2pix/results --input_nc 3 --output_nc 3 --preprocess none --dataset sharp2normal  --batch_size 25 --epoch latest  --netG resnet_6blocks```


## Best input configurations
|Architecture | Im2Im model name           |epoch |
| ------------|:--------------------------:|:----:|
| Cycle-GAN   | pgd2mnist                  | 7    |
| Cycle-GAN   | pgd2cifar10                |latest|
| Cycle-GAN   | pgd2Imagenet_no_norm_100   | 30   |
| pix2pix     | pix_pgd2mnist_resnet       |37    |
| pix2pix     | pix_pgd2Cifar10_resnet     |12    |
| pix2pix     | pix_spsa2cifar10           |11    |
| pix2pix     | pix_mix2Cifar10_resnet     |latest|
| pix2pix     | pix_fgs2cifar10            |latest|
| pix2pix     | pix_pgd2Imagenet_resnet    | 3    |
| pix2pix     | pix_pgd2Imagenet (U-net)   |latest|
| pix2pix     | pix_fgs2Imagenet_resnet    | 3    |
| pix2pix     | pix_dark2clear2            |latest|
| pix2pix     | pix_sharp2normal           |latest|
## training your own Im2Im translator

```cd CycleGAN_pix2pix```

### M=CycleGAN

```python new_train.py --dataroot ./[data-folder] --name [your-Im2Im-model] --model cycle_gan --use_wandb --save_epoch_freq 1 --dataset [dataset-name] --input_nc [=1 or =3] --output_nc [=1 or =3] --no_flip --preprocess None --attack [Attack-name] --eps [attack-size]```

Note: for ImageNet, [dataroot] = ./IMAGENET/train

### M=pix2pix

```python new_train.py --dataroot ./[data-folder] --model pix2pix --use_wandb --save_epoch_freq 1 --dataset [dataset-name] --eps [attack-size] --input_nc [=1 or =3] --output_nc [=1 or =3] --no_flip --preprocess None --name pix_[your-Im2Im-model] --attack [Attack-name] --netG resnet_6blocks```

Notes: 
- [Attack-name]=MIX to train a mix2cifar10 model.
- for ImageNet, [dataroot] = ./IMAGENET/train
