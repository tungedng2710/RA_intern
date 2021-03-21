from torch.utils.tensorboard import SummaryWriter
import torch
from preprocess.img_aug import ImageAugment
from dataset.dataloader import DatasetStatueHuman
import torch.optim as optim
import math
from models.model import HumanStatueModel
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from tqdm import *
import torch.utils.data as data
import errno
import time
import datetime
import os 
import random
import pandas as pd
from sklearn.metrics import confusion_matrix
# import torch.backends.cudnn as cudnn

def final_csv(csv_1, csv_2, num_select:tuple): 
    # Assume csv 2 is from WiderFace Additional Data
    # Num_select: (hm,st)
    df_1 = pd.read_csv(csv_1)
    df_2 = pd.read_csv(csv_2)

    df_2_human = df_2[df_2['label']==0]
    df_2_statue = df_2[df_2['label']==1]

    df_2_hm = df_2_human.sample(n=min(num_select[0], len(df_2_human)),replace=False)
    df_2_st = df_2_statue.sample(n=min(num_select[1], len(df_2_statue)),replace=False)

    df_final = pd.concat([df_1, df_2_hm, df_2_st], ignore_index=True, axis = 0)
    df_final.to_csv('/home/ducvuhong/statue_vs_human/csv/statue_vs_human/train/train_st_hm_final.csv', index=False)
    return

def train(dataloaders, scheduler, criterion, optimizer, epoch_init, batch_size, max_epoch=40):
    print('Begin')
    sigmoid = nn.Sequential(nn.Sigmoid())
    model.train()
    epoch = 0 + epoch_init
    best_val_acc = 0.0
    
    train_dataset = dataloaders["train"]
    val_dataset = dataloaders["val"]
    print("Data Train: ", len(train_dataset))
    print("Data Validation: ", len(val_dataset))
    
    train_iter_per_epoch = math.ceil(len(train_dataset) / batch_size)
    max_iter_train = max_epoch * train_iter_per_epoch
    
    val_iter_per_epoch = math.ceil(len(val_dataset) / batch_size)
    
    if epoch_init > 0:
        start_iter_train = epoch_init * train_iter_per_epoch
    else:
        start_iter_train = 0

    running_loss, running_corrects = 0.0, 0
    count, iter_n = 0, 0

    for iterate_num in tqdm(range(start_iter_train, max_iter_train)):
        if iterate_num % train_iter_per_epoch == 0:
            scheduler.step()
            train_batch_iter = iter(data.DataLoader(train_dataset, batch_size=32, shuffle=True))
            if epoch > 0:
                epoch_val_acc = val(epoch, 32, val_dataset, val_iter_per_epoch, sigmoid)
                if epoch_val_acc > best_val_acc:
                    print("\nBest Model Checkpoint found", epoch_val_acc)
                    ckpt_path = os.path.join(log_directory, "{}_{}.pth".format(epoch, epoch_val_acc))
                    torch.save(model.state_dict(), ckpt_path)
                    best_val_acc = epoch_val_acc
                model.train()
            epoch += 1
        
        time_start = time.time()

        # Load training data
        images, labels = next(train_batch_iter)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Zero-out optimizer
        optimizer.zero_grad()
        loss = criterion(outputs, labels.unsqueeze(1))

        preds = torch.round(sigmoid(outputs))

        loss.backward()
        optimizer.step()

        time_end = time.time()
        batch_time = time_end - time_start
        eta = int(batch_time * (max_iter_train - iterate_num))

        running_loss += loss.item() # Loss: Total loss for each iter 
        running_corrects += torch.sum(preds == labels.unsqueeze(1).data)

        # Log to Tensorboard in every n iters (100)
        if iterate_num % 100 == 0: 
            accuracy = running_corrects.double()/(100*batch_size)
            writer.add_scalar('Train Loss/Iter', running_loss/100, iterate_num + 1)
            writer.add_scalar('Train Accuracy', accuracy, iterate_num)
            # writer.add_scalar('LR', lr, iteration)
            print('[TRAINING]   Epoch:{}/{} - Learning rate: {:.4f} - Iter: {}/{} - Train Loss/Iter: {:.4f} - Train Accuracy: {:.4f} - Iteration Time: {:.4f}s - ETA: {}'
                .format(epoch, max_epoch, optimizer.param_groups[0]['lr'], iterate_num, max_iter_train, running_loss/100, accuracy, batch_time, str(datetime.timedelta(seconds=eta))))
            running_loss, running_corrects = 0.0, 0

@torch.no_grad()
def val(epoch, size_batch, val_dataset, max_iter, sigmoid):
    model.eval()
    print(max_iter)
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([3]).to(device))
    val_batch_iterator = iter(data.DataLoader(val_dataset, batch_size = size_batch, shuffle=True))
    run_loss, run_acc = 0.0, 0
    for images, labels in tqdm(val_batch_iterator):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images) 
        loss = criterion(outputs, labels.unsqueeze(1))
        preds = torch.round(sigmoid(outputs))
        run_loss += loss.item()
        run_acc += torch.sum(preds == labels.unsqueeze(1).data)
    
    accuracy = run_acc.double()/len(val_dataset)
    writer.add_scalar('Validation Loss/Iter', run_loss/max_iter, epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    print('[VALIDATING]   Epoch:{} - Validation Loss/Iter: {:.4f} - Validation Accuracy: {:.4f} - Length Dataset: {}'.format(epoch, run_loss/max_iter, accuracy, len(val_dataset)))
    return accuracy

if __name__ == '__main__':
    #model = HumanStatueModel(models.resnet18(pretrained = True), 2)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = models.resnet34(pretrained = True)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.25),
        nn.Linear(512, 1)
    )

    #model = model.load_state_dict(torch.load('/home/ducvuhong/statue_vs_human/train/log_dir/resnet_50/1_0.7883416458852868.pth', map_location = 'cpu'))
    model.to(device)
    loss = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([3]).to(device))
    opt = optim.Adam(model.fc.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer = opt, step_size = 10, gamma=0.75)

    ia_train = ImageAugment([False, True])
    ia_val = ImageAugment([False, False])

    toTensor = transforms.Compose([transforms.ToTensor()])
    dataloaders = {"train": DatasetStatueHuman(transform = ia_train, toTensor = toTensor , csv_file_ori =  '/home/ducvuhong/statue_vs_human/csv/fake3d_vs_human/train/train_3d_hm_final.csv'),
                "val": DatasetStatueHuman(transform = ia_val, toTensor = toTensor , csv_file_ori =  '/home/ducvuhong/statue_vs_human/csv/fake3d_vs_human/val/val_3d_hm_final.csv')
    }

    log_directory = '/home/ducvuhong/statue_vs_human/train/log_dir/resnet_34_19.03.21'
    try:
        os.makedirs(log_directory)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
    writer = SummaryWriter(log_dir = log_directory)

    train(dataloaders = dataloaders, scheduler = scheduler, criterion = loss, optimizer = opt, epoch_init = 0, max_epoch=45, batch_size = 32)
    