

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os import path
import os
import matplotlib.pyplot as plt

mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_loaders(args):
    args.mu = mu
    args.std = std
    valdir = path.join(args.data_dir, 'val')
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize(args.img_size),
                                                           transforms.CenterCrop(args.crop_size),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=args.mu, std=args.std)
                                                           ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=args.workers, pin_memory=True)
    return val_loader