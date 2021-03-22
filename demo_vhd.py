import json
import torchvision.models as models
import torch
from PIL import Image
import pandas as pd 
import os
import sys

def chip_list(json_data):
        chip = []
        for img in json_data:
            for img_chip in img['image_chips']:
                chip.append(img_chip)
        return chip

def change_coordinate(coef, box):
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





if __name__ == '__main__':
    json_path = "data.json"
    f = open(json_path, "w")
    data = json.load(f) 

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_cls = load_model("vhd_resnet34", "/Users/mac/Projects/MergeRetina/RA_intern/8_0.9138495092693566.pth")



    json.dump(data, f)
    f.close()


# def crop_full(self):
#         _create_folder(self.output_statue)
#         _create_folder(self.output_human)
#         chip = self._chip_list()
#         num_statue, num_face = 0, 0

#         csv_file = open(self.csv_path, mode='a')
#         fieldnames = ['image_path', 'data', 'label']
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         if self.append_mode == False:
#             writer.writeheader()

#         with tqdm(total=len(chip)) as progressbar:
#             for img_chip in chip:
#                 img_path = os.path.join(self.root, img_chip['chip_name'])
#                 for bbox in img_chip['chip_valid_bboxes']:
#                     box = (bbox['bbox_x1'], bbox['bbox_x2'], bbox['bbox_y1'], bbox['bbox_y2'])
#                     if bbox['class'] == 3 or bbox['class'] == 2 or bbox['class'] == 5: # Fake 2D - No MR - Human
#                         continue
#                     elif bbox['class'] == 1: # MR - Face
#                         path_output = os.path.join(self.output_human, f'{img_chip["chip_name"][:-4]}_{img_chip["chip_id"]}_{bbox["bbox_x1"]}_{bbox["bbox_x2"]}.jpg')
#                         self.crop_save_img(img_path, path_output, box)
#                         writer.writerow({'image_path': path_output, 'data': self.dataset, 'label': 0}) # Human
#                         num_face +=1
#                     else:
#                         path_output = os.path.join(self.output_statue, f'{img_chip["chip_name"][:-4]}_{img_chip["chip_id"]}_{bbox["bbox_x1"]}_{bbox["bbox_x2"]}.jpg')
#                         self.crop_save_img(img_path, path_output, box)
#                         writer.writerow({'image_path': path_output, 'data': self.dataset, 'label': 1}) # Statue
#                         num_statue +=1
#                 progressbar.update(1)
#                 progressbar.set_postfix({"Number of Statues": num_statue, "Number of Faces": num_face})
#         csv_file.close()