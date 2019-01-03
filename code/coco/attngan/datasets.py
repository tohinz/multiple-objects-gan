from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
from PIL import Image
import numpy.random as random

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from miscc.utils import *

def prepare_data(data, eval=False):
    if eval:
        imgs, captions, captions_lens, class_ids, keys, transformation_matrices, label, bbox = data
    else:
        imgs, captions, captions_lens, class_ids, keys, transformation_matrices, label = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    transformation_matrices[0] = transformation_matrices[0][sorted_cap_indices]
    transformation_matrices[1] = transformation_matrices[1][sorted_cap_indices]
    label = label[sorted_cap_indices]
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        transformation_matrices[0] = transformation_matrices[0].cuda()
        transformation_matrices[1] = transformation_matrices[1].cuda()
        label = label.cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    if eval:
        return [real_imgs, captions, sorted_cap_lens, class_ids, keys, transformation_matrices, label, bbox]
    else:
        return [real_imgs, captions, sorted_cap_lens, class_ids, keys, transformation_matrices, label]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)

    img, bbox_scaled = crop_imgs(img, bbox)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.ToPILImage()(img)
                re_img = transforms.Resize((imsize[i], imsize[i]))(re_img)
            else:
                re_img = transforms.ToPILImage()(img)
            ret.append(normalize(re_img))

    return ret, bbox_scaled


def crop_imgs(image, bbox, max_objects=3):
    ori_size = 268
    imsize = 256

    flip_img = random.random() < 0.5
    img_crop = ori_size - imsize
    h1 = int(np.floor((img_crop) * np.random.random()))
    w1 = int(np.floor((img_crop) * np.random.random()))

    bbox_scaled = np.zeros_like(bbox)
    bbox_scaled[...] = -1.0

    for idx in range(max_objects):
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

    cropped_image = image[:, w1: w1 + imsize, h1: h1 + imsize]

    if flip_img:
        idx = [i for i in reversed(range(cropped_image.shape[2]))]
        idx = torch.LongTensor(idx)
        transformed_image = torch.index_select(cropped_image, 2, idx)
    else:
        transformed_image = cropped_image

    return transformed_image, bbox_scaled


class TextDataset(data.Dataset):
    def __init__(self, data_dir, img_dir, split='train', base_size=64,
                 transform=None, target_transform=None, eval=False):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.img_dir = img_dir
        self.split_dir = os.path.join(data_dir, split)
        self.eval = eval

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = self.load_bbox()
        self.labels = self.load_labels()
        self.split_dir = os.path.join(data_dir, split)
        self.max_objects = 3

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(self.split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        bbox_path = os.path.join(self.split_dir, 'bboxes.pickle')
        with open(bbox_path, "rb") as f:
            bboxes = pickle.load(f)
            bboxes = np.array(bboxes)
        print("bboxes: ", bboxes.shape)
        return bboxes

    def load_labels(self):
        label_path = os.path.join(self.split_dir, 'labels.pickle')
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
            labels = np.array(labels)
        print("labels: ", labels.shape)
        return labels

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_transformation_matrices(self, bbox):
        bbox = torch.from_numpy(bbox)
        bbox = bbox.view(-1, 4)
        transf_matrices_inv = compute_transformation_matrix_inverse(bbox)
        transf_matrices_inv = transf_matrices_inv.view(self.max_objects, 2, 3)
        transf_matrices = compute_transformation_matrix(bbox)
        transf_matrices = transf_matrices.view(self.max_objects, 2, 3)

        return transf_matrices, transf_matrices_inv

    def get_one_hot_labels(self, label):
        labels = torch.from_numpy(label)
        labels = labels.long()
        # remove -1 to enable one-hot converting
        labels[labels < 0] = 80
        label_one_hot = torch.FloatTensor(labels.shape[0], 81).fill_(0)
        label_one_hot = label_one_hot.scatter_(1, labels, 1).float()

        return label_one_hot

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[index]

        img_name = '%s/%s.jpg' % (self.img_dir, key)
        imgs, bbox_scaled = get_imgs(img_name, self.imsize,
                            bbox, self.transform, normalize=self.norm)
        transformation_matrices = self.get_transformation_matrices(bbox_scaled)

        # load label
        label = self.labels[index]
        label = self.get_one_hot_labels(label)

        # randomly select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        if self.eval:
            return imgs, caps, cap_len, cls_id, key, transformation_matrices, label, bbox_scaled
        return imgs, caps, cap_len, cls_id, key, transformation_matrices, label

    def __len__(self):
        return len(self.filenames)
