import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, AvgPool2d, AdaptiveAvgPool2d, Linear, Softmax
import torchvision.models as models


resnet34 = models.resnet34()
resnet50 = models.resnet50()
resnet101 = models.resnet101()
vgg19mod = models.vgg19()


class ResNet34mod(nn.Module):
    def __init__(self):
        super(ResNet34mod, self).__init__()
        self.backbone = resnet101
        self.fc1 = nn.Linear(in_features = 1000, out_features = 512, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(in_features = 512, out_features = 2, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x


class ResNet50mod(nn.Module):
    def __init__(self):
        super(ResNet50mod, self).__init__()
        self.backbone = resnet50

        self.fc1 = nn.Linear(in_features = 1000, out_features = 512, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(in_features = 512, out_features = 2, bias=True)
        self.sm = Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x
        
class ResNet101mod(nn.Module):
    def __init__(self):
        super(ResNet101mod, self).__init__()
        self.backbone = resnet101

        self.fc1 = nn.Linear(in_features = 1000, out_features = 512, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(in_features = 512, out_features = 512, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.fc3 = nn.Linear(in_features = 512, out_features = 2, bias=True)


    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.dropout2(x)
        x = self.fc3(x)

        return x