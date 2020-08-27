import torch
from torchvision import datasets, transforms
import random
from .rotation import RotateImageFolder
from pdb import set_trace as st


def get_cifa10(train, split, batch_size, image_size):


    if train:

      transform = transforms.Compose([
          transforms.RandomCrop(image_size, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])


    else: 
      transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])


    if split is '0':
      knownclass = [0, 1 ,2 ,4 ,5 ,9]
      unknownclass = [3, 6, 7, 8]
    elif split is '1':
      knownclass = [0, 3 ,5 ,7 ,8 ,9]
      unknownclass = [1, 2, 4, 6]
    elif split is '2':
      knownclass = [0, 1 ,5 ,6, 7 ,8]
      unknownclass = [2, 3, 4, 9]
    elif split is '3':
      knownclass = [3, 4 ,5 ,7, 8 ,9]
      unknownclass = [0, 1, 2, 6]
    elif split is '4':
      knownclass = [0, 1 ,2 ,3, 7 ,8]
      unknownclass = [4, 5, 6, 9]
    elif split is '10':
      knownclass = [0, 1 ,2 ,3, 4, 5, 6, 7 ,8, 9]
      unknownclass = []


    dataset = datasets.CIFAR10(root='../datasets/cifar10', 
                              download=True, 
                              train=train,
                              transform=transform)



    kmask = [i for i,e  in enumerate(dataset) if e[1] in knownclass]
    unkmask = [i for i,e  in enumerate(dataset) if e[1] in unknownclass]

    if train:

        random.shuffle(kmask)
        validationportion = int(0.1*len(kmask))

        kmask_rand_val = kmask[:validationportion]
        kmask_rand_train= kmask[validationportion:]

        known_set_train = torch.utils.data.Subset(dataset, kmask_rand_train)
        known_set_val = torch.utils.data.Subset(dataset, kmask_rand_val)

        known_set_train = RotateImageFolder(known_set_train)

        known_data_loader_train = torch.utils.data.DataLoader(
            dataset=known_set_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

        known_data_loader_val = torch.utils.data.DataLoader(
            dataset=known_set_val,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

        return known_data_loader_train, known_data_loader_val, knownclass

    else:

        known_set = torch.utils.data.Subset(dataset, kmask)
        unknown_set = torch.utils.data.Subset(dataset, unkmask)

        known_data_loader = torch.utils.data.DataLoader(
            dataset=known_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

        unknown_data_loader = torch.utils.data.DataLoader(
            dataset=unknown_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

        return known_data_loader, unknown_data_loader, knownclass