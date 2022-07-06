from .base_options import BaseOptions
import numpy as np

class TestOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        
        parser.add_argument('--dataset', default='MNIST', help='dataset use to train the classifier')
        parser.add_argument('--dataset_test', default='MNIST', help='dataset of test input')
        parser.add_argument('--num_workers', type=str, default=2, help='number of workers for data loaders')
        parser.add_argument('--attack', type=str, default='PGD', help='testing attack, could be (PGD,FGS, CW or NoAttack')
        parser.add_argument('--eps', type=float, default=0.3, help='perturbation size')
        parser.add_argument('--attack_norm', type=float, default=np.inf, help='perturbation size')
        parser.add_argument('--defense', type=str, default='Im2Im', help='defense can be [Im2Im, adv_train]')
        
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser




















'''
#parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
#parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#parser.add_argument('--checkpoints_dir', type=str, default='CycleGAN_pix2pix/checkpoints', help='(M) models are saved here')
#parser.add_argument('--batch_size', type=str, default=128, help='batch size for data loaders')


parser.add_argument('--M_name', type=str, default='pgd2mnist', help='used pre-classification module M')
parser.add_argument('--M_method', type=str, default='cycleGAN', help='used pre-classification method for M (cycleGAN, pix2pix, etc')
'''
        