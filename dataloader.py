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
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.Resize(self.input_shape[1:]),
                transforms.ToTensor(),
                #torch.from_numpy(transforms)
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        label_idx = self.label[index]
        img = Image.open(image_path)
        img = self.transform(img)
        return img, label_idx


if __name__ == '__main__':
    dataset = ReviewAssisstantDataset(csv_file = '/home/ducvuhong/statue_vs_human/data/train.csv', 
                            root_dir='/home/ducvuhong/statue_vs_human/data/train/human/', 
                            input_shape = (3, 224, 224),
                            mode = 'train'
                            )
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    sample, label = dataset.__getitem__(1)
    print(sample)