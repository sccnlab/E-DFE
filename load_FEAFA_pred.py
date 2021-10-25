#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:25:01 2020
@author: craigposkanzer
"""

import os, time
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
from nest import register, Context
from typing import Tuple, List, Dict, Union, Callable, Optional




class Image_Dataset(Dataset):
    """AU image dataset."""
    def __init__(self, data=[], transform = None):
        'Initialization'
        #self.split = split
        self.data = []
        self.transform = transform
    def get_train(self, img_dir):
        pass         

    def get_test(self, img_dir):
        folder = sorted(os.listdir(img_dir))
        for i in range(0, len(folder)):
            image_x = img_dir + folder[i]
            self.data.append(image_x)
                
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image  = self.data[idx]
        if self.transform:
           img = self.transform(Image.open(image))
        else:
           img = io.imread(image)
        return img
    
    
@register
def loadfeafa_pred(
    split: str,
    data_dir: str,
    label_path: Optional[str] = None,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None) -> Dataset:
    '''MakeHuman Images.
    '''
    if split == 'train':
        trainDataset = Image_Dataset(transform = transform)
        trainDataset.get_train(data_dir)
        return trainDataset
    if split == 'test':
        testDataset = Image_Dataset(transform = transform)
        testDataset.get_test(data_dir)
        return testDataset


