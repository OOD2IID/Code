# -*- coding: utf-8 -*-

from CycleGAN_pix2pix.models.cycle_gan_model import CycleGANModel
from CycleGAN_pix2pix.options.test_options import TestOptions
#from CycleGAN_pix2pix.models import create_model

import os
import torch
from PIL import Image
import numpy as np
# Get current working directory
cwd = os.getcwd()

class WrappedModel(torch.nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module
	def forward(self, x):
		return self.module(x)
    

def load_M(M_name, M_method,device,checkpoint_dir):
    '''load pre-classification model M'''
    
    if M_method =='cycleGAN':
        opt = TestOptions().parse()
        #M = create_model(opt)      # create a model given opt.model and other options
        #M.setup(opt)
        #M.eval()
        M = CycleGANModel(opt).netG_A
        #M.to(device)
        
        if isinstance(M, torch.nn.DataParallel):
            M = M.module
        # original saved file with DataParallel
        load_path = os.path.join(cwd,checkpoint_dir,M_name,'latest_net_G_A.pth')
        
        state_dict = torch.load(load_path,map_location=str(device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        
        
        keys = state_dict.keys()
        for key in list(keys):  # need to copy keys here because we mutate in loop
            patch_instance_norm_state_dict(state_dict, M, key.split('.'))
            #print(type(state_dict))
        
        M.load_state_dict(state_dict,strict=False)
        
        return M
    
def patch_instance_norm_state_dict(state_dict, module, keys,i=0):
    """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
    key = keys[i]
    if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
        if module.__class__.__name__.startswith('InstanceNorm') and \
                (key == 'running_mean' or key == 'running_var'):
            if getattr(module, key) is None:
                state_dict.pop('.'.join(keys))
        if module.__class__.__name__.startswith('InstanceNorm') and \
           (key == 'num_batches_tracked'):
            state_dict.pop('.'.join(keys))
            
        #return state_dict
    else:
        #print(i)
        
        patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    

        
    

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)    



def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    
    
    #image_pil = Image.fromarray((255*image_numpy.permute(1, 2, 0)).numpy().astype(np.uint8))
    image_pil = Image.fromarray((image_numpy).astype(np.uint8))
    
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)
