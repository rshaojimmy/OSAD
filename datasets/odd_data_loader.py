import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import Sampler
import torchvision
import numpy as np
import random
from torch.utils.data import *


class RelblDataset(Dataset):

    def __init__(self, dataset, cname, classname=-1):
        self.dataset = dataset
        self.cname = cname
        if classname == -1:
            self.mask = [1] * len(dataset)
        else:
            self.mask = [i for i, e in enumerate(dataset) if e[1] == classname]

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        return (self.dataset[idx][0], self.cname)



class OCSampler(Sampler):

    def __init__(self, mask):
        self.mask = mask	

    def __iter__(self):
        return (iter(self.mask))

    def __len__(self):
        return len(self.mask)



def get_loader_odd(classname=[0,1,2,3,4,5,6,7,8,9], ds='OOD', split=None ):
    data_transforms = {
     'train': transforms.Compose([  transforms.Resize(32),  transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768])
     ]),
     'val': transforms.Compose([transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
     ]),
     }
    if ds == 'CIFAR10':
        mnist = datasets.CIFAR10(root='../data/cifar', download=True, transform=data_transforms['train'] )
        mnist_test = datasets.CIFAR10(root='../data/cifar', train=False, download=True, transform=data_transforms['val'])
    elif ds == 'CIFAR100':
        mnist = datasets.CIFAR100(root='../data/cifar100', download=True, transform=data_transforms['train'] )
        mnist_test = datasets.CIFAR100(root='../data/cifar100', train=False, download=True, transform=data_transforms['val'])
    if ds == 'STL10':
        mnist = datasets.STL10(root='../data/STL', download=True, transform=data_transforms['train'] )
        mnist_test = datasets.STL10(root='../data/STL', train=False, download=True, transform=data_transforms['val'])
    if ds == 'SVHN':
        mnist = datasets.SVHN('../data/svhn', split='train', download=True, transform=data_transforms['train'] )
        mnist_test = datasets.SVHN('../data/svhn', split='test',  download=True, transform=data_transforms['val'])
    if ds == 'OOD':
        if split == 0:
           neg = torchvision.datasets.ImageFolder(root='../Imagenet', transform=data_transforms['val'])
        elif split == 1:
           neg = torchvision.datasets.ImageFolder(root='../Imagenet_resize', transform=data_transforms['val'])
        elif split == 2:
           neg = torchvision.datasets.ImageFolder(root='../LSUN', transform=data_transforms['val'])
        elif split == 3:
           neg = torchvision.datasets.ImageFolder(root='../LSUN_resize', transform=data_transforms['val'])
        print(split)
        print(len(neg))
        neg = RelblDataset(neg, 10)
        mnist = datasets.CIFAR10(root='../data/cifar', download=True, transform=data_transforms['train'] )
        mnist_test = datasets.CIFAR10(root='../data/cifar', train=False, download=True, transform=data_transforms['val'])
        mnist_test = torch.utils.data.ConcatDataset((mnist_test,neg))


    mask = [i for i,e  in enumerate(mnist) if e[1] in classname]
    manualSeed = 0
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.shuffle(mask)
    train_mask = mask[0:int(len(mask)*0.9)]
    val_mask = mask[int(len(mask)*0.9):int(len(mask))]
    MNISTTrainsampler = OCSampler(train_mask)
    MNISTValSampler = OCSampler(val_mask)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=128, sampler=MNISTTrainsampler, num_workers=1)
    mnist_val_loader = torch.utils.data.DataLoader(dataset=mnist,sampler=MNISTValSampler,
                                               batch_size=128,num_workers=1)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test, 
                                               batch_size=128, num_workers= 1, shuffle=True)
    return  mnist_loader, mnist_val_loader, mnist_test_loader


