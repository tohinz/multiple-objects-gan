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
    def __init__(self, data_dir, img_dir, imsize, split='train', embedding_type='cnn-rnn', transform=None, crop=True, stage=1):

        self.transform = transform
        self.imsize = imsize
        self.crop = crop
        self.data = []
        self.data_dir = data_dir
        self.split_dir = os.path.join(data_dir, split)
        self.img_dir = img_dir
        self.max_objects = 3
        self.stage = stage

        self.filenames = self.load_filenames()
        self.bboxes = self.load_bboxes()
        self.labels = self.load_labels()
        self.embeddings = self.load_embedding(self.split_dir, embedding_type)

    def get_img(self, img_path):
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

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

    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().decode('utf8').split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            embeddings = np.array(embeddings)
        return embeddings

    def load_filenames(self):
        filepath = os.path.join(self.split_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def crop_imgs(self, image, bbox):
        ori_size = image.shape[1]
        imsize = self.imsize

        flip_img = random.random() < 0.5
        img_crop = ori_size - self.imsize
        h1 = int(np.floor((img_crop) * np.random.random()))
        w1 = int(np.floor((img_crop) * np.random.random()))

        if self.stage == 1:
            bbox_scaled = np.zeros_like(bbox)
            bbox_scaled[...] = -1.0

            for idx in range(self.max_objects):
                bbox_tmp = bbox[idx]
                if bbox_tmp[0] == -1:
                    break

                x_new = max(bbox_tmp[0] * float(ori_size) - h1, 0) / float(imsize)
                y_new = max(bbox_tmp[1] * float(ori_size) - w1, 0) / float(imsize)

                width_new = min((float(ori_size)/imsize) * bbox_tmp[2], 1.0)
                if x_new + width_new > 0.999:
                    width_new = 1.0 - x_new - 0.001

                height_new = min((float(ori_size)/imsize) * bbox_tmp[3], 1.0)
                if y_new + height_new > 0.999:
                    height_new = 1.0 - y_new - 0.001

                if flip_img:
                    x_new = 1.0-x_new-width_new

                bbox_scaled[idx] = [x_new, y_new, width_new, height_new]
        else:
            # need two bboxes for stage 1 G and stage 2 G
            bbox_scaled = [np.zeros_like(bbox), np.zeros_like(bbox)]
            bbox_scaled[0][...] = -1.0
            bbox_scaled[1][...] = -1.0

            for idx in range(self.max_objects):
                bbox_tmp = bbox[idx]
                if bbox_tmp[0] == -1:
                    break

                # scale bboxes for stage 1 G
                stage1_size = 64
                stage1_ori_size = 76
                x_new = max(bbox_tmp[0] * float(stage1_ori_size) - h1, 0) / float(stage1_size)
                y_new = max(bbox_tmp[1] * float(stage1_ori_size) - w1, 0) / float(stage1_size)

                width_new = min((float(stage1_ori_size) / stage1_size) * bbox_tmp[2], 1.0)
                if x_new + width_new > 0.999:
                    width_new = 1.0 - x_new - 0.001

                height_new = min((float(stage1_ori_size) / stage1_size) * bbox_tmp[3], 1.0)
                if y_new + height_new > 0.999:
                    height_new = 1.0 - y_new - 0.001

                if flip_img:
                    x_new = 1.0 - x_new - width_new

                bbox_scaled[0][idx] = [x_new, y_new, width_new, height_new]

                # scale bboxes for stage 2 G
                x_new = max(bbox_tmp[0] * float(ori_size) - h1, 0) / float(imsize)
                y_new = max(bbox_tmp[1] * float(ori_size) - w1, 0) / float(imsize)

                width_new = min((float(ori_size) / imsize) * bbox_tmp[2], 1.0)
                if x_new + width_new > 0.999:
                    width_new = 1.0 - x_new - 0.001

                height_new = min((float(ori_size) / imsize) * bbox_tmp[3], 1.0)
                if y_new + height_new > 0.999:
                    height_new = 1.0 - y_new - 0.001

                if flip_img:
                    x_new = 1.0 - x_new - width_new

                bbox_scaled[1][idx] = [x_new, y_new, width_new, height_new]


        cropped_image = image[:, w1: w1 + imsize, h1: h1 + imsize]

        if flip_img:
            idx = [i for i in reversed(range(cropped_image.shape[2]))]
            idx = torch.LongTensor(idx)
            transformed_image = torch.index_select(cropped_image, 2, idx)
        else:
            transformed_image = cropped_image

        return transformed_image, bbox_scaled


    def __getitem__(self, index):
        # load image
        key = self.filenames[index]
        img_name = self.img_dir +"/" + key + ".jpg"
        img = self.get_img(img_name)

        # load bbox
        bbox = self.bboxes[index]

        # load label
        label = self.labels[index]

        # load caption embedding
        embeddings = self.embeddings[index, :, :]
        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]

        if self.crop:
            img, bbox = self.crop_imgs(img, bbox)

        return img, bbox, label, embedding

    def __len__(self):
        return len(self.filenames)
