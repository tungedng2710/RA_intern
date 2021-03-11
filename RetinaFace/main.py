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

parser = argparse.ArgumentParser()
parser.add_argument('--cls_model', type=str, default = '/home/tungnguyen/Review/cls_models/resnet50_40epochs.pth', help = 'path to classify model')
parser.add_argument('--cuda', type=str, default = '0', help = 'use gpu if it is available')

args = parser.parse_args()

device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")


# print("start")
# subprocess.call("/home/tungnguyen/Review/RetinaFace/infer.sh", shell = True)
# print("end")

def get_samples(images_folder = None, json_file = None):
    if images_folder == None:
        images_folder = '/home/tungnguyen/Review/RetinaFace/photos'
    if json_file == None:
        json_file = '/home/tungnguyen/Review/RetinaFace/results/prediction_demo.json'
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
        images_folder = '/home/tungnguyen/Review/RetinaFace/photos'
    path_to_model = args.cls_model
    if path_to_model.find('resnet') != -1:

        model_name = 'ResNet-50'
        model = models.resnet50()
        model.to(device)
        model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
        model.eval()

    elif path_to_model.find('vgg') != -1:
        model_name = 'VGG-19'
        model = models.vgg19()
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
        # key = samples['names'][i]
        # final_output.append(
        #     {
        #         key: []
        #     }
        # )
        num_human_face = 0
        num_statue_face = 0
        img = images[i]
        print('Process ', samples['names'][i] + '.jpg')
        print('Iamge shape:', img.shape)

        for box in samples['bboxes'][i]:
            x1 = box[0]
            x2 = box[1]
            y1 = box[2]
            y2 = box[3]

            sample = img[y1:y2, x1:x2, :]
            cv2.imwrite('bgr.jpg', sample)
            cropped_img = cv2.imread('bgr.jpg')
            cropped_img = cv2.resize(cropped_img, dsize=(112, 112), interpolation = cv2.INTER_CUBIC)

            #prediction
            sm = torch.nn.Softmax()
            test = torch.from_numpy(np.moveaxis(cropped_img, -1, 0))
            output = model((test[None, ...]/255).to(device))
            probabilities = sm(output) 
            
            _, pred = torch.max(output, 1)

            #visualization
            thickness = 2
            if pred[0] == 0:
                num_human_face = num_human_face + 1
                color = (0, 255, 0)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                #cv2.putText(img, str(torch.max(probabilities)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            elif pred[0] == 1:
                num_statue_face = num_statue_face + 1
                color = (255, 0, 0)
                img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                #cv2.putText(img, str(torch.max(probabilities)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # print('Number of human face: ', num_human_face)
        # print('Number of statue face:', num_statue_face)
        cv2.imwrite('results/' + samples['names'][i] + '.jpg', img)


if __name__ == '__main__':
    start = timeit.default_timer()
    print('RetinaFace is running...')
    #subprocess.call("/home/tungnguyen/Review/RetinaFace/infer.sh", shell = True)
    print('=====================================================================')
    print('Classifier is running...')
    run()
    stop = timeit.default_timer()

    print('Total time: ', (stop - start))  
