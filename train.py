import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import copy
import time
import numpy as np
import pickle
import argparse
import os.path as osp
from collections import OrderedDict
from misc.utils import mkdir
from models import *
from misc.utils import init_model, init_random_seed, mkdirs, lab_conv
from torch.nn import DataParallel
from models import DenoiseResnet
from advertorch.context import ctx_noparamgrad_and_eval
import torchvision.utils as vutils

from misc.saver import Saver
from datasets.dataloader import get_data_loader

from pdb import set_trace as st




def train_Ours(args, train_loader, val_loader, knownclass, Encoder, Decoder, NorClsfier, SSDClsfier, summary_writer, saver):
    seed = init_random_seed(args.manual_seed)

    criterionCls = nn.CrossEntropyLoss()
    criterionRec = nn.MSELoss()

    if args.parallel_train:
        Encoder = DataParallel(Encoder)    
        Decoder = DataParallel(Decoder)    
        NorClsfier = DataParallel(NorClsfier)    
        SSDClsfier = DataParallel(SSDClsfier)    
    
    optimizer = optim.Adam(list(Encoder.parameters())
                          +list(NorClsfier.parameters())
                          +list(SSDClsfier.parameters())
                          +list(Decoder.parameters()), lr=args.lr)
 


    if args.adv is 'PGDattack':
        print("**********Defense PGD Attack**********")
    elif args.adv is 'FGSMattack':
        print("**********Defense FGSM Attack**********")

    if args.adv is 'PGDattack':
        from advertorch.attacks import PGDAttack
        nor_adversary = PGDAttack(predict1=Encoder, predict2=NorClsfier, nb_iter=args.adv_iter)
        rot_adversary = PGDAttack(predict1=Encoder, predict2=SSDClsfier, nb_iter=args.adv_iter)

    elif args.adv is 'FGSMattack':
        from advertorch.attacks import GradientSignAttack
        nor_adversary = GradientSignAttack(predict1=Encoder, predict2=NorClsfier)
        rot_adversary = GradientSignAttack(predict1=Encoder, predict2=SSDClsfier)

    global_step = 0
    # ----------
    #  Training
    # ----------
    for epoch in range(args.n_epoch):
        
        Encoder.train()
        Decoder.train()        
        NorClsfier.train()        
        SSDClsfier.train()        
   
        for steps, (orig, label, rot_orig, rot_label) in enumerate(train_loader):

            label = lab_conv(knownclass, label)
            orig, label = orig.cuda(), label.long().cuda()

            rot_orig, rot_label = rot_orig.cuda(), rot_label.long().cuda()

            with ctx_noparamgrad_and_eval(Encoder):
                with ctx_noparamgrad_and_eval(NorClsfier):
                    with ctx_noparamgrad_and_eval(SSDClsfier):
                        adv = nor_adversary.perturb(orig, label)
                        rot_adv = rot_adversary.perturb(rot_orig, rot_label) 

            latent_feat = Encoder(adv)
            norpred =  NorClsfier(latent_feat)
            norlossCls = criterionCls(norpred, label)

            recon = Decoder(latent_feat)
            lossRec = criterionRec(recon, orig)

            ssdpred = SSDClsfier(Encoder(rot_adv))
            rotlossCls = criterionCls(ssdpred, rot_label)

            loss = args.norClsWgt*norlossCls + args.rotClsWgt*rotlossCls + args.RecWgt*lossRec


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            #============ tensorboard the log info ============#
            lossinfo = {
                'loss': loss.item(),               
                'norlossCls': norlossCls.item(), 
                'lossRec': lossRec.item(),                                                                                     
                'rotlossCls': rotlossCls.item(),                                                                                     
                    } 


            global_step+=1
   
            #============ print the log info ============# 
            if (steps+1) % args.log_step == 0:
                errors = OrderedDict([('loss', loss.item()),
                                    ('norlossCls', norlossCls.item()),
                                    ('lossRec', lossRec.item()),
                                    ('rotlossCls', rotlossCls.item()),
                                        ]) 
              
                saver.print_current_errors((epoch+1), (steps+1), errors) 


        # evaluate performance on validation set periodically
        if ((epoch + 1) % args.val_epoch == 0):

            # switch model to evaluation mode
            Encoder.eval()
            NorClsfier.eval()

            running_corrects = 0.0
            epoch_size = 0.0
            val_loss_list = []

            # calculate accuracy on validation set
            for steps, (images, label) in enumerate(val_loader):

                label = lab_conv(knownclass, label)
                images, label = images.cuda(), label.long().cuda()
    
                adv = nor_adversary.perturb(images, label)

                with torch.no_grad():
                    logits = NorClsfier(Encoder(adv))
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == label.data)
                    epoch_size += images.size(0)
                    
                    val_loss = criterionCls(logits, label)

                    val_loss_list.append(val_loss.item())

            val_loss_mean = sum(val_loss_list)/len(val_loss_list)

            val_acc =  running_corrects.double() / epoch_size
            print('Val Acc: {:.4f}, Val Loss: {:.4f}'.format(val_acc, val_loss_mean))

            valinfo = {
                'Val Acc': val_acc.item(),               
                'Val Loss': val_loss.item(), 
                    } 
            for tag, value in valinfo.items():
                summary_writer.add_scalar(tag, value, (epoch + 1))

            orig_show = vutils.make_grid(orig, normalize=True, scale_each=True)
            recon_show = vutils.make_grid(recon, normalize=True, scale_each=True)

            summary_writer.add_image('Ori_Image', orig_show, (epoch + 1))
            summary_writer.add_image('Rec_Image', recon_show, (epoch + 1))

        

        if ((epoch + 1) % args.model_save_epoch == 0):
            model_save_path = os.path.join(args.results_path, args.training_type, 
                            'snapshots', args.datasetname+'-'+args.split, args.denoisemean, 
                             args.adv+str(args.adv_iter))    
            mkdir(model_save_path) 
            torch.save(Encoder.state_dict(), os.path.join(model_save_path,
                "Encoder-{}.pt".format(epoch+1)))
            torch.save(NorClsfier.state_dict(), os.path.join(model_save_path,
                "NorClsfier-{}.pt".format(epoch+1)))
            torch.save(Decoder.state_dict(), os.path.join(model_save_path,
                "Decoder-{}.pt".format(epoch+1)))



    torch.save(Encoder.state_dict(), os.path.join(model_save_path, "Encoder-final.pt"))
    torch.save(NorClsfier.state_dict(), os.path.join(model_save_path, "NorClsfier-final.pt"))
    torch.save(Decoder.state_dict(), os.path.join(model_save_path, "Decoder-final.pt"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="AdvOpenset")
    
    parser.add_argument('--adv', type=str, default='FGSMattack') #clean PGDattack FGSMattack
    parser.add_argument('--adv_iter', type=int, default=5)

    parser.add_argument('--results_path', type=str, default='./results/')
    parser.add_argument('--training_type', type=str, default='Ours_FD')

    parser.add_argument('--parallel_train', type=str, default=False) # cifar10 svhn False; tinyimagenet True 
    parser.add_argument('--datasetname', type=str, default='cifar10') # cifar10 tinyimagenet svhn
    parser.add_argument('--split', type=str, default='0')
    parser.add_argument('--img_size', type=int, default=32)  # cifar svhn 32 tinyimagenet 64
    
    parser.add_argument('--denoisemean', type=str, default='gaussian')
    parser.add_argument('--init_type', type=str, default='normal') # normal xavier kaiming

    parser.add_argument('--denoise', type=str, default=[True, True, True, True, True]) 
    parser.add_argument('--norClsWgt', type=int, default=1)
    parser.add_argument('--rotClsWgt', type=int, default=1) 
    parser.add_argument('--RecWgt', type=int, default=1)
    parser.add_argument('--latent_size', type=int, default=512)

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--n_epoch', type=int, default=120) # cifar svhn 120 tinyimagenet 150
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--model_save_epoch', type=int, default=1)
    parser.add_argument('--val_epoch', type=int, default=1)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--manual_seed', type=int, default=None)

    args = parser.parse_args()
    print(args)

    log_file = os.path.join(args.results_path, args.training_type, 'log', 
        args.datasetname+'-'+args.split, args.denoisemean, args.adv+str(args.adv_iter))

    
    summary_writer = SummaryWriter(log_file)
    saver = Saver(args, log_file)
    saver.print_config()

    train_loader, val_loader, knownclass = get_data_loader(name=args.datasetname, train=True, split=args.split, 
                                    batch_size=args.batchsize, image_size=args.img_size)

    nclass = len(knownclass)

    Encoderrestore = None     
    Decoderrestore = None     
    NorClsfierrestore = None     
    SSDClsfierrestore = None     


    Encoder = init_model(net=DenoiseResnet.ResnetEncoder(denoisemean=args.denoisemean, 
                    latent_size= args.latent_size, denoise=args.denoise), 
                    init_type = args.init_type, restore=Encoderrestore)
    Decoder = init_model(net=DenoiseResnet.ResnetDecoder(latent_size= args.latent_size), 
                    init_type = args.init_type, restore=Decoderrestore)

    NorClsfier = init_model(net=DenoiseResnet.NorClassifier(latent_size= args.latent_size, num_classes=nclass), 
                    init_type = args.init_type, restore=NorClsfierrestore)

    SSDClsfier = init_model(net=DenoiseResnet.SSDClassifier(latent_size= args.latent_size), 
                    init_type = args.init_type, restore=SSDClsfierrestore)

    Encoder.init_nonlocal()


    train_Ours(args, train_loader, val_loader, knownclass, Encoder, Decoder, NorClsfier, SSDClsfier, summary_writer, saver)


