import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            dense_sizes = [(min_size / self.image_size[0], min_size / self.image_size[1]) for min_size in min_sizes]
            dense_ratio = [self.steps[k] / self.image_size[0], self.steps[k] / self.image_size[1]]
            cf = [
                [(c + 0.5) * dense_ratio[0] for c in range(f[0])],
                [(c + 0.5) * dense_ratio[1] for c in range(f[1])],
            ]

            count = 0
            for cy, cx in product(cf[0], cf[1]):
                for ds_y, ds_x in dense_sizes:
                    anchors += [cx, cy, ds_x, ds_y]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output