import cv2
import json
import matplotlib.pyplot as plt 
import numpy as np
import os
import sys
import glob
from tqdm import *
import argparse
import subprocess
from PIL import Image
import torch
import timeit
import torchvision.models as models
from models import *
import torch.nn.functional as nnf

parser = argparse.ArgumentParser()
parser.add_argument('--cls_model', type=str, default = '/home/tungnguyen/Review/cls_models/resnet101mod_40epochs_best_model.pth', help = 'path to classify model')
parser.add_argument('--cuda', type=str, default = '0', help = 'use gpu if it is available')
parser.add_argument('--expand_box', type=float, default = '0.2', help = 'use gpu if it is available')
parser.add_argument('--input_shape', type=int, default = 224, help = 'shape of input image')

args = parser.parse_args()

device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")


# print("start")
# subprocess.call("/home/tungnguyen/Review/RetinaFace/infer.sh", shell = True)
# print("end")

def get_samples(images_folder = None, json_file = None):
    if images_folder == None:
        images_folder = '/home/tungnguyen/Review/photos'
    if json_file == None:
        json_file = '/home/tungnguyen/Review/results/prediction_demo.json'
    path2train_json = json_file


    with open(json_file) as jfile:
        statue_widerface_train = json.load(jfile)
    image_names = list(statue_widerface_train.keys())

    data = {
        'names': [],
        'bboxes': []
    }
    for name in image_names:
        bboxes = []
        for pred in statue_widerface_train[name]['prediction']:
            x1 = int(pred['pred_bbox_x1'])
            x2 = int(pred['pred_bbox_x2'])
            y1 = int(pred['pred_bbox_y1'])
            y2 = int(pred['pred_bbox_y2'])
            box = [x1, x2, y1, y2]
            bboxes.append(box)
            #print(box)
        data['names'].append(name)
        data['bboxes'].append(bboxes)

    return data, statue_widerface_train


def run(images_folder = None):
    final_output = []

    if images_folder == None:
        images_folder = '/home/tungnguyen/Review/photos'
    path_to_model = args.cls_model
    if path_to_model.find('resnet50mod') != -1:
        model_name = 'ResNet-50mod'
        model = ResNet50mod()
        model.to(device)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        model.eval()
    elif path_to_model.find('resnet101mod') != -1:
        model_name = 'ResNet-101mod'
        model = ResNet101mod()
        model.to(device)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        model.eval()

    elif path_to_model.find('vgg19') != -1:
        model_name = 'VGG-19'
        model = models.vgg19()
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        model.to(device)
        model.eval()
    else:
        model_name = 'DucNet'
        model = models.resnet34(pretrained = True)
        model.fc = nn.Sequential(
                            nn.BatchNorm1d(512),
                            nn.Dropout(p=0.25),
                            nn.Linear(512, 1)
                        )
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        model.to(device)
        model.eval()

    print('Classifier is using ' + model_name + '...')

    samples, json_file = get_samples()
    cropped_images = []
    images = []

    for i in range (len(samples['names'])):
        images.append(cv2.imread(images_folder + '/' + samples['names'][i] + '.jpg'))
    for i in tqdm(range (len(samples['names']))):
        img = images[i]
        # print('Process ', samples['names'][i] + '.jpg')
        # print('Image shape:', img.shape)

        for box in samples['bboxes'][i]:
            x1 = box[0]
            x2 = box[1]
            y1 = box[2]
            y2 = box[3]
            img_shape = img.shape
            y_length, x_length = y2 - y1, x2 - x1
            x_change_side = x_length*args.expand_box
            y_change_side = y_length*args.expand_box
            ymin, ymax = y1 - y_change_side, y2 + y_change_side
            xmin, xmax = x1 - x_change_side, x2 + x_change_side
            new_box = (ymin, ymax, xmin, xmax)
            y_start, y_end, x_start, x_end = map(int, map(round, new_box))
            y_start = max(y_start, 0)
            y_end = min(y_end, img_shape[0])
            x_start = max(x_start, 0)
            x_end = min(x_end, img_shape[1])

            sample = img[y_start:y_end, x_start:x_end, :]
            # sample = img[y1:y2, x1:x2, :]
            # Load via PIL
            cv2.imwrite('bgr.jpg', sample)
            cropped_img = cv2.imread('bgr.jpg')
            dshape = args.input_shape
            cropped_img = cv2.resize(cropped_img, dsize=(dshape, dshape), interpolation = cv2.INTER_AREA)
            os.remove('bgr.jpg')


            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            cropped_img = cropped_img/255
            cropped_img[:, :, 0] = (cropped_img[:, :, 0] - mean[2])/std[2]
            cropped_img[:, :, 1] = (cropped_img[:, :, 1] - mean[1])/std[1]
            cropped_img[:, :, 2] = (cropped_img[:, :, 2] - mean[0])/std[0]

            #prediction
            

            sm = torch.nn.Softmax()
            test = torch.from_numpy(np.moveaxis(cropped_img, -1, 0))
            output = model((test[None, ...]).to(device, dtype=torch.float))

            # sigmoid = nn.Sequential(nn.Sigmoid())
            # pred = torch.round(sigmoid(output))
            # print(pred.cpu().data.numpy())
            probabilities = sm(output) 
            _, pred = torch.max(output, 1)
            conf = torch.max(probabilities).cpu().data.numpy()
            
            #visualization
            thickness = 2
            if pred[0] == 0:
                #num_human_face = num_human_face + 1
                color = (0, 255, 0)
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(img, 'human '+ str(conf), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            elif pred[0] == 1:
                #num_statue_face = num_statue_face + 1
                color = (255, 0, 0)
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                cv2.putText(img, 'statue ' + str(conf), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            
            else:
                pass
        cv2.imwrite('results/' + samples['names'][i] + '_'+model_name+'.jpg', img)


if __name__ == '__main__':
    start = timeit.default_timer()
    print('RetinaFace is running...')
    #subprocess.call("/home/tungnguyen/Review/RetinaFace/infer.sh", shell = True)
    print('=====================================================================')
    print('Classifier is running...')
    run()
    stop = timeit.default_timer()

    print('Total time: ', (stop - start))  
