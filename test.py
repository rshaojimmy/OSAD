import os
import os.path as osp
import argparse

import torch
from torch import nn
from tensorboardX import SummaryWriter
from datasets.dataloader import get_data_loader
from misc.utils import init_model, init_random_seed, mkdirs
from misc.saver import Saver
import random
from models import *
from models import DenoiseResnet
from OpensetMethods.OpenMax import openmax

from pdb import set_trace as st

def test(args):

    args.seed = init_random_seed(args.manual_seed)


    #####################load datasets##################### 

    kdataloader_trn,_, knownclass  = get_data_loader(name=args.datasetname, train=True, split=args.split, 
                                    batch_size=args.batchsize, image_size=args.imgsize)

    kdataloader_tst, ukdataloader_tst, knownclass = get_data_loader(name=args.datasetname, train=False, split=args.split, 
                                    batch_size=args.batchsize, image_size=args.imgsize)
    
    nclass = len(knownclass)    
    #####################Network Init##################### 
    Encoderrestore = osp.join('results', args.defense, 'snapshots', 
                args.datasetname+'-'+args.split, args.denoisemean, args.adv+str(args.adv_iter), 'Encoder-'+args.defensesnapshot+'.pt') 
    Encoder = init_model(net=DenoiseResnet.ResnetEncoder(denoisemean=args.denoisemean, 
                    latent_size= args.latent_size, denoise=args.denoise), 
                    init_type = args.init_type, restore=Encoderrestore, parallel_reload=args.parallel_train)        


    NorClsfierrestore = osp.join('results', args.defense, 'snapshots', 
            args.datasetname+'-'+args.split, args.denoisemean, args.adv+str(args.adv_iter), 'NorClsfier-'+args.defensesnapshot+'.pt')
    NorClsfier = init_model(net=DenoiseResnet.NorClassifier(latent_size= args.latent_size, num_classes=nclass), 
                init_type = args.init_type, restore=NorClsfierrestore, parallel_reload=args.parallel_train)



    openmax(args, kdataloader_trn, kdataloader_tst, ukdataloader_tst, knownclass, Encoder, NorClsfier)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="AdvOpenset")

    parser.add_argument('--training_type', type=str, default='Test')
    parser.add_argument('--parallel_train', type=str, default=False) # cifar10 svhn False; tinyimagenet True
    parser.add_argument('--datasetname', type=str, default='cifar10') # cifar10 tinyimagenet svhn     
    parser.add_argument('--split', type=str, default='0') 
    parser.add_argument('--imgsize', type=int, default=32) # cifar svhn 32 tinyimagenet 64

    parser.add_argument('--adv', type=str, default='FGSMattack') #clean PGDattack FGSMattack 
    parser.add_argument('--adv_iter', type=int, default=5)

    parser.add_argument('--defense', type=str, default='Ours_FD') 
    parser.add_argument('--denoisemean', type=str, default='gaussian') 
    parser.add_argument('--init_type', type=str, default='normal')

    parser.add_argument('--defensesnapshot', type=str, default='1') 
    parser.add_argument('--denoise', type=str, default=[True, True, True, True, True]) 

    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=512)

    parser.add_argument('--results_path', type=str, default='./results/')
    parser.add_argument('--manual_seed', type=int, default=None)



    print(parser.parse_args())
    test(parser.parse_args())

