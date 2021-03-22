import PIL 
import numpy as np
from imgaug import augmenters as iaa 
from torchvision import transforms

class ImageAugment:
    def __init__(self, augment_img):
        self.augment_img = augment_img
        self.statue = iaa.Sequential([
            iaa.Scale((224,224)),
			iaa.Fliplr(0.5),
            iaa.AddToHueAndSaturation((-20,20)),
			iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))),
			iaa.Sometimes(0.8, iaa.OneOf([ # Geometric Transformations
                    iaa.Affine(rotate=(-45, 45)),
                    iaa.Affine(shear=(-20, 20))
            ])),
			iaa.Sometimes(0.3, iaa.GaussianBlur((0, 0.15))), # blur images with a sigma between 0 and 0.25
			iaa.Sometimes(0.5, iaa.OneOf([
                    iaa.GammaContrast((0.5, 2.0)),
                    iaa.LinearContrast((0.4, 1.5))
            ])),
			iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
        ])
        self.human = iaa.Sequential([
            iaa.Scale((224,224)),
			iaa.Fliplr(0.5),
            iaa.Sometimes(0.4, iaa.OneOf([
                                    iaa.AddToHueAndSaturation((-15, 15)),
                                    iaa.Add((-10, 10), per_channel=0.5),
                                    iaa.Multiply((0.75, 1.5), per_channel=0.5),
                                    iaa.GaussianBlur((0, 0.5)),
                                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                                    iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3))
                                    ])
                        )
                ])  
        self.norm = iaa.Sequential([
            iaa.Scale((224,224)),
			iaa.Fliplr(0.5)
        ])
            
    def __call__(self, img, label = 0):
        img = np.array(img)
        if label == 1:
            if self.augment_img[1]:
                return self.statue.augment_image(img)
            else: 
                return self.norm.augment_image(img)
        else:
            if self.augment_img[0]:
                return self.human.augment_image(img)
            else: 
                return self.norm.augment_image(img)