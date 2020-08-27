from __future__ import division

import os
import random
from PIL import Image
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn import init
import scipy.io as scio
import math
from torch.nn import DataParallel
from torch import nn
from copy import deepcopy
from advertorch.context import ctx_noparamgrad_and_eval

from pdb import set_trace as st


def make_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().detach().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(imtype)


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def weights_init_const(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.constant_(m.weight.data, 1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)



def weights_init_normal(m):

    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if (m.bias is not None) and (m.bias.data is not None):
            m.bias.data.zero_()
    elif (classname.find('BatchNorm') != -1):
        if (m.weight is not None) and (m.weight.data is not None):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
    elif (classname.find('Linear') != -1):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

        

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal_rnn(m):
    classname = m.__class__.__name__
    if classname.find('LSTM') != -1:
        init.orthogonal_(m.all_weights[0][0], gain=1)
        init.orthogonal_(m.all_weights[0][1], gain=1)
        init.constant_(m.all_weights[0][2], 1)
        init.constant_(m.all_weights[0][3], 1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type == 'orthogonal_rnn':
        net.apply(weights_init_orthogonal_rnn)
    elif init_type == 'const':
        net.apply(weights_init_const)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)                


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed




def init_model(net, restore, init_type, init= True, parallel_reload=True):
    """Init models with cuda and weights."""
    # init weights of model
    if init:
        init_weights(net, init_type)
    else:
        print("inside normalization")
    
    # restore model weights
    if restore is not None and os.path.exists(restore):

        # original saved file with DataParallel
        state_dict = torch.load(restore)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if parallel_reload:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        
        net.restored = True
        print("*************Restore model from: {}".format(os.path.abspath(restore)))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)




def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil = image_pil.resize((128, 128), resample=Image.BICUBIC)
    image_pil.save(image_path)


def lab_conv(knownclass, label):
    knownclass = sorted(knownclass)
    
    label_convert = torch.zeros(len(label))

    for j in range(len(label)):
        for i in range(len(knownclass)):
            if label[j] == knownclass[i]:
                label_convert[j] = int(knownclass.index(knownclass[i]))

    return label_convert


