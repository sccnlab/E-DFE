#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
        exp_folder = sorted(os.listdir(img_dir))
        for identity in range(1, 26, 2):
            id_folder = sorted(os.listdir(os.path.join(img_dir, exp_folder[identity])))            
            for video in range(1,len(id_folder)-2):
                video_folder = sorted(os.listdir(os.path.join(img_dir, exp_folder[identity], id_folder[video]))) 
                for frame in range(0,len(video_folder),2):
                    image_x = os.path.join(img_dir, exp_folder[identity], id_folder[video], video_folder[frame +1])                      
                    output_i = os.path.join(img_dir, exp_folder[identity], id_folder[video], video_folder[frame])                   
                    self.data.append((image_x, output_i))
            

    def get_test(self, img_dir):
        exp_folder = sorted(os.listdir(img_dir))
        for identity in range(1,len(exp_folder)-1,2):
            id_folder = sorted(os.listdir(os.path.join(img_dir, exp_folder[identity]))) 
            for video in range(len(id_folder)-2, len(id_folder)):
                video_folder = sorted(os.listdir(os.path.join(img_dir, exp_folder[identity], id_folder[video])))
                for frame in range(0,len(video_folder),2):
                    image_x = os.path.join(img_dir, exp_folder[identity], id_folder[video], video_folder[frame +1])                       
                    output_i = os.path.join(img_dir, exp_folder[identity], id_folder[video], video_folder[frame])                  
                    self.data.append((image_x, output_i))

             
                
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image, label  = self.data[idx]
        if self.transform:
           img = self.transform(Image.open(image))
        else:
           img = io.imread(image)
        output_i_raw = open(label)
        output_i = np.genfromtxt(output_i_raw)
        target = torch.tensor(output_i).type(torch.FloatTensor)
        return img, target
    
    
@register
def loadfeafa(
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


