import json
import torchvision.models as models
import torch
from PIL import Image
import pandas as pd 
import os
import sys
from tqdm import *
from preprocess.img_aug import ImageAugment
import torch.nn as nn 

def chip_list(json_data):
        chip = []
        for img in json_data:
            for img_chip in img['image_chips']:
                chip.append(img_chip)
        return chip

def change_coordinate(coef, box : tuple):
    x1, y1, x2, y2 = box
    if coef == 0:
        return x1, y1, x2, y2 = map(int, map(round, box))
    else:
        x_length, y_length = x2-x1, y2-y1
        x_change, y_change = 1/2*x_length*coef, 1/2*y_length*coef
        x1, y1, x2, y2 = x1-x_change, y1-y_change, x2+x_change, y2+y_change
        new_box = (x1, y1, x2, y2)
        return x1, y1, x2, y2 = map(int, map(round, new_box))

def load_model(model_name : str, ckpt_path):
    if model_name == 'resnet50mod'
        model = ResNet50mod()
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.eval()
    elif model_name == 'resnet101mod':
        model = ResNet101mod()
        model.to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.eval()
    elif model_name == 'vgg19':
        model = models.vgg19()
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(device)
        model.eval()
    else: # vhd model branch
        model = models.resnet34(pretrained = False)
        model.fc = nn.Sequential(
                            nn.BatchNorm1d(512),
                            nn.Dropout(p=0.25),
                            nn.Linear(512, 1)
                        )
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.to(device)
        model.eval()
    return model

def main(data_json, root_dir, sig, model_cls):
    ia = ImageAugment([False, False])
    image_list = data_json.keys()
    for img in image_list:
        img_path = os.path.join(root_dir, img + '.jpg')
        for pred in data[i]['prediction']:
            x1, y1, x2, y2 = change_coordinate(0.9, (pred['pred_bbox_x1'], pred['pred_bbox_y1'], pred['pred_bbox_x2'], pred['pred_bbox_y2']))
            image = Image.open(img_path)
            bbox_img = image[y1:y2, x1:x2]
            bbox_img = torch.Tensor(bbox_img)
            bbox_img = torch.unsqueeze(bbox_img, 0)
            output = int(torch.round(sigmoid(model(bbox_img))))
            if output == 0:
                pred['class'] = 'human'
            else:
                pred['class'] = 'statue'
    return data_json
            
if __name__ == '__main__':
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_cls = load_model("vhd_resnet34", "/Users/mac/Projects/MergeRetina/RA_intern/8_0.9138495092693566.pth")
    json_path = "data.json"
    f = open(json_path, "w")
    data = json.load(f) 
    root_dir = '/home/nguyenhuuminh/MyProjects/ReviewAssistant/pvn-review-assistant/data/pixta_22_43/test/' 
    sigmoid = nn.Sequential(
        nn.Sigmoid()
    )
    data_new = main(data, root_dir, sigmoid, model_cls)
    json.dump(data, f)
    f.close()
    