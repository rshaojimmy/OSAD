import torch
from torchvision import datasets, transforms
import random
from torchvision.datasets import ImageFolder
import numpy as np
from pdb import set_trace as st
from .rotation import RotateImageFolder


def get_tinyimagenet(train, split, batch_size, image_size):


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



    dataset=ImageFolder(root='../datasets/TinyImageNet', 
                        transform=transform)  


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    torch.manual_seed(20200221)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    
    if split is '0':
      rand_allclass = np.random.RandomState(seed=20200221).permutation(len(dataset.classes)).tolist()
    elif split is '1':
      rand_allclass = np.random.RandomState(seed=20200231).permutation(len(dataset.classes)).tolist()
    elif split is '2':
      rand_allclass = np.random.RandomState(seed=20200241).permutation(len(dataset.classes)).tolist()

    knownclass = rand_allclass[:20]
    unknownclass = rand_allclass[20:]

    if train:

        kmask = [i for i,e  in enumerate(train_dataset) if e[1] in knownclass]
        unkmask = [i for i,e  in enumerate(train_dataset) if e[1] in unknownclass]

        random.shuffle(kmask)
        validationportion = int(0.1*len(kmask))

        kmask_rand_val = kmask[:validationportion]
        kmask_rand_train= kmask[validationportion:]

        known_set_train = torch.utils.data.Subset(train_dataset, kmask_rand_train)
        known_set_val = torch.utils.data.Subset(train_dataset, kmask_rand_val)

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

        kmask = [i for i,e  in enumerate(test_dataset) if e[1] in knownclass]
        unkmask = [i for i,e  in enumerate(test_dataset) if e[1] in unknownclass]

        known_set = torch.utils.data.Subset(test_dataset, kmask)
        unknown_set = torch.utils.data.Subset(test_dataset, unkmask)


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