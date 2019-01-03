from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.utils.data as data
import PIL
import os
import os.path
import random
import numpy as np
import json
import glob
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from miscc.config import cfg
from miscc.utils import *

shape_dict = {
    "cube": 0,
    "cylinder": 1,
    "sphere": 2
}

color_dict  = {
    "gray": 0,
    "red": 1,
    "blue": 2,
    "green": 3,
    "brown": 4,
    "purple": 5,
    "cyan": 6,
    "yellow": 7
}


class TextDataset(data.Dataset):
    def __init__(self, data_dir, imsize, split='train', transform=None):

        self.transform = transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.img_dir = os.path.join(self.split_dir, "images")
        self.scene_dir = os.path.join(self.split_dir, "scenes")
        self.max_objects = 4

        self.filenames = self.load_filenames()

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        flip_img = random.random() < 0.5
        if flip_img:
            idx = [i for i in reversed(range(img.shape[2]))]
            idx = torch.LongTensor(idx)
            img = torch.index_select(img, 2, idx)

        return img, flip_img

    def load_bboxes(self):
        bbox_path = os.path.join(self.split_dir, 'bboxes.pickle')
        with open(bbox_path, "rb") as f:
            bboxes = pickle.load(f)
            bboxes = np.array(bboxes)
        return bboxes

    def load_labels(self):
        label_path = os.path.join(self.split_dir, 'labels.pickle')
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            labels = np.array(labels)
        return labels

    def load_filenames(self):
        filenames = [filename for filename in glob.glob(self.scene_dir + '/*.json')]
        print('Load scenes from: %s (%d)' % (self.scene_dir, len(filenames)))
        return filenames

    def calc_transformation_matrix(self, bbox):
        bbox = torch.from_numpy(bbox)
        bbox = bbox.view(-1, 4)
        transf_matrices_inv = compute_transformation_matrix_inverse(bbox)
        transf_matrices_inv = transf_matrices_inv.view(self.max_objects, 2, 3)
        transf_matrices = compute_transformation_matrix(bbox)
        transf_matrices = transf_matrices.view(self.max_objects, 2, 3)
        return transf_matrices, transf_matrices_inv

    def label_one_hot(self, label, dim):
        labels = torch.from_numpy(label)
        labels = labels.long()
        # remove -1 to enable one-hot converting
        labels[labels < 0] = dim-1
        label_one_hot = torch.FloatTensor(labels.shape[0], dim).fill_(0)
        label_one_hot = label_one_hot.scatter_(1, labels, 1).float()
        return label_one_hot

    def __getitem__(self, index):
        # load image
        key = self.filenames[index]
        with open(key, "rb") as f:
            json_file = json.load(f)
        img_name = self.img_dir +"/" + json_file["image_filename"]
        img, flip_img = self.get_img(img_name)

        # load bbox#
        bbox = np.zeros((self.max_objects, 4), dtype=np.float32)
        bbox[:] = -1.0
        for idx in range(len(json_file["objects"])):
            bbox[idx, :] = json_file["objects"][idx]["bbox"]
        bbox = bbox / float(self.imsize)

        # load label
        # shapes: 3; colors: 8; materials: 2 (not used), size: 3 (but given through bbox)
        label_shape = np.zeros(self.max_objects)
        label_color = np.zeros(self.max_objects)
        label_shape[:] = -1
        label_color[:] = -1
        for idx in range(len(json_file["objects"])):
            label_shape[idx] = shape_dict[json_file["objects"][idx]["shape"]]
            label_color[idx] = color_dict[json_file["objects"][idx]["color"]]

        label_shape = self.label_one_hot(np.expand_dims(label_shape, 1), 4)
        label_color = self.label_one_hot(np.expand_dims(label_color, 1), 9)
        label = torch.cat((label_shape, label_color), 1)

        if flip_img:
            bbox[:, 0] = 1.0 - bbox[:, 0] - bbox[:, 2]
        transformation_matrices = self.calc_transformation_matrix(bbox)

        return img, transformation_matrices, label, bbox

    def __len__(self):
        return len(self.filenames)
