import json
import cv2
import os
import tqdm
import errno
import math
from tqdm import *
import csv

def _create_folder(folder_path):
    try:
        os.makedirs(folder_path)
    except OSError as exc:  # Race condition
        if exc.errno != errno.EEXIST:
            raise


class CropImage:
    def __init__(self, root, data, percent_area_inc, output_statue, output_human, csv_path, append_mode, dataset):
        self.data = data
        self.percent_area_inc = percent_area_inc
        self.output_statue = output_statue
        self.output_human = output_human
        self.root = root
        self.csv_path = csv_path
        self.append_mode = append_mode
        self.dataset = dataset

    def _chip_list(self):
        chip = []
        for img in self.data:
            for img_chip in img['image_chips']:
                chip.append(img_chip)
        return chip
    
    def _area(self, box):
        x1, y1, x2, y2 = box
        x_length, y_length = x2 - x1, y2 - y1
        return x_length*y_length
    
    def _change_coordinate(self, box):
        x1, y1, x2, y2 = box
        if self.percent_area_inc == 0:
            x1, y1, x2, y2 = map(int, map(round, box))
            return x1, x2, y1, y2
        else: # Revision later 
            rate_change = math.sqrt(1+self.percent_area_inc)-1
            x_length, y_length = x2 - x1, y2 - y1
            x_change, y_change = 1/2*x_length*rate_change, 1/2*y_length*rate_change
            x_start = x1 - x_change
            x_end = x2 + x_change
            y_start = y1 - y_change
            y_end = y2 + y_change
            box_adj = (x_start, x_end, y_start, y_end)
            xstart, xend, ystart, yend = map(int, map(round, box_adj))
            return xstart, xend, ystart, yend
    
    def crop_save_img(self, img_path, path_output, box):
        img_ori = cv2.imread(img_path)
        xstart, xend, ystart, yend = self._change_coordinate(box)
        img_crop = img_ori[xend:yend, xstart:ystart]
        cv2.imwrite(path_output, img_crop)

    def crop_full(self):
        _create_folder(self.output_statue)
        _create_folder(self.output_human)
        chip = self._chip_list()
        num_statue, num_face = 0, 0
        
        csv_file = open(self.csv_path, mode='a')
        fieldnames = ['image_path', 'data', 'label']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if self.append_mode == False:
            writer.writeheader()

        with tqdm(total=len(chip)) as progressbar:
            for img_chip in chip:
                img_path = os.path.join(self.root, img_chip['chip_name'])
                for bbox in img_chip['chip_valid_bboxes']:
                    box = (bbox['bbox_x1'], bbox['bbox_x2'], bbox['bbox_y1'], bbox['bbox_y2'])
                    if bbox['class'] == 3 or bbox['class'] == 2 or bbox['class'] == 5: # Fake 2D - No MR - Human
                        continue
                    elif bbox['class'] == 1: # MR - Face
                        path_output = os.path.join(self.output_human, f'{img_chip["chip_name"][:-4]}_{img_chip["chip_id"]}_{bbox["bbox_x1"]}_{bbox["bbox_x2"]}.jpg')
                        self.crop_save_img(img_path, path_output, box)
                        writer.writerow({'image_path': path_output, 'data': self.dataset, 'label': 0}) # Human
                        num_face +=1
                    else:
                        path_output = os.path.join(self.output_statue, f'{img_chip["chip_name"][:-4]}_{img_chip["chip_id"]}_{bbox["bbox_x1"]}_{bbox["bbox_x2"]}.jpg')
                        self.crop_save_img(img_path, path_output, box)
                        writer.writerow({'image_path': path_output, 'data': self.dataset, 'label': 1}) # Statue
                        num_statue +=1
                progressbar.update(1)
                progressbar.set_postfix({"Number of Statues": num_statue, "Number of Faces": num_face})
        csv_file.close()

def run():
    json_path = '/home/nguyenhuuminh/MyProjects/ReviewAssistant/pvn-review-assistant/data/pixta_22_43_statue_widerface/statue_val.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    root_path = '/home/nguyenhuuminh/MyProjects/ReviewAssistant/pvn-review-assistant/data/pixta_22_43_statue_widerface/images'
    statue_path = '/home/tungnguyen/Review/data/val'
    human_path = '/home/tungnguyen/Review/data/val'
    csv_path = '/home/tungnguyen/Review/data/val2.csv'
    crop_img = CropImage(root_path, data, 0, statue_path, human_path, csv_path, True, 'WiderFace')
    crop_img.crop_full()

if __name__ == '__main__':
    run()