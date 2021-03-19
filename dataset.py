from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class ReviewAssisstantDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, input_shape, mode):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.input_shape = input_shape
        self.mode = mode

        # Load csv and info file
        data_list = pd.read_csv(csv_file)
        self.image_path_list = data_list['image_path'].to_list()
        self.label = data_list['label'].to_list()

        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.input_shape[1:]),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.Resize(self.input_shape[1:]),
                transforms.ToTensor(),
                #torch.from_numpy(transforms)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label_idx = self.label[index]
        img = Image.open(image_path)
        img = self.transform(img)
        return img, label_idx


class DatasetStatueHuman(Dataset):
    def __init__(self, toTensor, transform, csv_file_ori, input_shape = (3, 224, 224)):
        self.transform = transform
        self.csv_file_ori = csv_file_ori
        self.input_shape = input_shape       

        # Load csv and info file
        if type(csv_file_ori) == pd.core.frame.DataFrame:
            data_list_ori = csv_file_ori
        else:
            data_list_ori = pd.read_csv(csv_file_ori)
        # data_list_ori = pd.read_csv(csv_file_ori)
        
        self.image_path_list = data_list_ori['image_path'].to_list()
        self.label = data_list_ori['label'].to_list()      
        self.toTensor = toTensor
    
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label_idx = float(self.label[index])
        img = Image.open(image_path)
        img = self.transform(img, label_idx)
        img = self.toTensor(img)
        return img, label_idx
    
    def __len__(self):
        return len(self.label)