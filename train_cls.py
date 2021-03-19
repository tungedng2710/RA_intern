import cv2
import json
import numpy as np
import os
import sys
import glob
import argparse
import time
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, AvgPool2d, AdaptiveAvgPool2d, Linear
import torchvision.models as models
import torch.optim as optim

from dataloader import ReviewAssisstantDataset


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = 'resnet50', help = 'resnet50, vgg19')
parser.add_argument('--cuda', type=str, default = '0', help = 'use gpu if it is available')
parser.add_argument('--train_set', type=str, default = None, help = 'path to train dataset')
parser.add_argument('--val_set', type=str, default = None, help = 'path to validation dataset')

parser.add_argument('--batch_size', type=int, default = 64, help = 'batch_size for training')
parser.add_argument('--num_epochs', type=int, default = 20, help = 'number of epochs')
parser.add_argument('--lr', type=float, default = 0.0001, help = 'learning rate')

args = parser.parse_args()


def train_model(model, dataloaders, criterion, optimizer, 
                num_epochs=args.num_epochs, is_inception=False):

    device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")
    model.to(device)
    since = time.time()
    lr = args.lr 
    val_acc_history = []
    losses = []

    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            lr = lr / 2
            print("learning rate: ", lr)
        optimizer = optim.Adam(model.parameters(), lr)
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                losses.append(epoch_loss)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    torch.save(model.state_dict(), 'cls_models/' + args.model + '_' + str(args.num_epochs) + 'epochs' + '.pth')

    torch.cuda.empty_cache()
    return model, val_acc_history, losses

def run():
    train_set = ReviewAssisstantDataset(csv_file = '/home/ducvuhong/statue_vs_human/data/train_pixta.csv', 
                            root_dir='/home/tungnguyen/Review/data/statue_vs_human/data/train/full', 
                            input_shape = (3, 224, 224),
                            mode = 'train'
                            )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=True, num_workers=8)

    val_set = ReviewAssisstantDataset(csv_file = '/home/tungnguyen/Review/data/val.csv', 
                            root_dir='/home/tungnguyen/Review/data/val', 
                            input_shape = (3, 224, 224),
                            mode = 'val'
                            )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = args.batch_size, shuffle=True, num_workers=8)

    dataloader = {
        'train': train_loader,
        'val': val_loader
    }

    model_zoo = ['resnet50', 'vgg19']
    if args.model == 'resnet50':
        model = models.resnet50()
    if args.model == 'vgg19':
        model = models.vgg19()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    model, val_acc_history, losses = train_model(model, dataloader, criterion, optimizer, 
                                                    num_epochs = args.num_epochs, 
                                                    is_inception=False)


if __name__ == '__main__':
    run()
    
