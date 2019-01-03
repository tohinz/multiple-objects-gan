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
from PIL import Image
import torchvision.transforms as transforms
import torch
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from miscc.config import cfg


class TextDataset(data.Dataset):
    def __init__(self, data_dir, imsize, split='train', transform=None, crop=False):

        self.transform = transform
        self.imsize = imsize
        self.crop = crop
        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split, "normal")
        self.img_dir = self.split_dir + "/imgs/"
        self.max_objects = 3

        self.filenames = self.load_filenames()
        self.bboxes = self.load_bboxes()
        self.labels = self.load_labels()

    def get_img(self, img_path):
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def load_bboxes(self):
        bbox_path = os.path.join(self.split_dir, 'bboxes.pickle')
        with open(bbox_path, "rb") as f:
            bboxes = pickle.load(f)
            bboxes = np.array(bboxes, dtype=np.double)
        return bboxes

    def load_labels(self):
        label_path = os.path.join(self.split_dir, 'labels.pickle')
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            labels = np.array(labels)
        return labels

    def load_filenames(self):
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        # load image
        key = self.filenames[index]
        key = key.split("/")[-1]
        img_name = self.split_dir + "/imgs/" + key
        img = self.get_img(img_name)

        # load bbox
        bbox = self.bboxes[index].astype(np.double)

        # load label
        label = self.labels[index]

        return img, bbox, label

    def __len__(self):
        return len(self.filenames)
