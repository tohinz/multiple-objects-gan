import os
import errno
import numpy as np
import cPickle as pickle
import glob

from copy import deepcopy
from miscc.config import cfg

from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import grad
from torch.autograd import Variable


def compute_transformation_matrix_inverse(bbox):
    x, y = bbox[:, 0], bbox[:, 1]
    w, h = bbox[:, 2], bbox[:, 3]

    scale_x = 1.0 / w
    scale_y = 1.0 / h

    t_x = 2 * scale_x * (0.5 - (x + 0.5 * w))
    t_y = 2 * scale_y * (0.5 - (y + 0.5 * h))

    zeros = torch.cuda.DoubleTensor(bbox.shape[0],1).fill_(0)

    transformation_matrix = torch.cat([scale_x.unsqueeze(-1), zeros, t_x.unsqueeze(-1),
                                       zeros, scale_y.unsqueeze(-1), t_y.unsqueeze(-1)], 1).view(-1, 2, 3)

    return transformation_matrix


def compute_transformation_matrix(bbox):
    x, y = bbox[:, 0], bbox[:, 1]
    w, h = bbox[:, 2], bbox[:, 3]

    scale_x = w
    scale_y = h

    t_x = 2 * ((x + 0.5 * w) - 0.5)
    t_y = 2 * ((y + 0.5 * h) - 0.5)

    zeros = torch.cuda.DoubleTensor(bbox.shape[0],1).fill_(0)

    transformation_matrix = torch.cat([scale_x.unsqueeze(-1), zeros, t_x.unsqueeze(-1),
                                       zeros, scale_y.unsqueeze(-1), t_y.unsqueeze(-1)], 1).view(-1, 2, 3)

    return transformation_matrix


def pad_imgs(img, pad=2):
    m = nn.ConstantPad2d((pad, pad, pad, pad), 0)
    return m(img)


def load_validation_data(datapath):
    with open(os.path.join(datapath, "normal", "bboxes.pickle"), "rb") as f:
        bboxes = pickle.load(f)
        bboxes = np.array(bboxes)

    with open(os.path.join(datapath, "normal", "labels.pickle"), "rb") as f:
        labels = pickle.load(f)
        labels = np.array(labels)

    return torch.from_numpy(labels), torch.from_numpy(bboxes)


def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               local_label, transf_matrices, transf_matrices_inv, gpus):
    criterion = nn.BCEWithLogitsLoss()
    batch_size = real_imgs.size(0)
    fake = fake_imgs.detach()
    local_label = local_label.detach()
    local_label_cond = local_label[:, 0, :] + local_label[:, 1, :] + local_label[:, 2, :]
    real_features = nn.parallel.data_parallel(netD, (real_imgs, local_label, transf_matrices, transf_matrices_inv), gpus)
    fake_features = nn.parallel.data_parallel(netD, (fake, local_label, transf_matrices, transf_matrices_inv), gpus)
    # real pairs
    inputs = (real_features, local_label_cond)
    real_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_real = criterion(real_logits, real_labels)
    # wrong pairs
    inputs = (real_features[:(batch_size-1)], local_label_cond[1:])
    wrong_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake pairs
    inputs = (fake_features, local_label_cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = nn.parallel.data_parallel(netD.get_uncond_logits, (real_features), gpus)
        fake_logits = nn.parallel.data_parallel(netD.get_uncond_logits, (fake_features), gpus)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        #
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()


def compute_generator_loss(netD, fake_imgs, real_labels, local_label, transf_matrices, transf_matrices_inv, gpus):
    criterion = nn.BCEWithLogitsLoss()
    local_label = local_label.detach()
    local_label_cond = local_label[:, 0, :] + local_label[:, 1, :] + local_label[:, 2, :]
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs, local_label, transf_matrices, transf_matrices_inv), gpus)
    # fake pairs
    inputs = (fake_features, local_label_cond)
    fake_logits = nn.parallel.data_parallel(netD.get_cond_logits, inputs, gpus)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = nn.parallel.data_parallel(netD.get_uncond_logits, (fake_features), gpus)
        # fake_logits = torch.clamp(fake_logits, 1e-8, 1-1e-8)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    # data_img is changed to [0,1]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples.png' % image_dir,
            normalize=True)
        # fake.data is still [-1, 1]
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)


def save_model(netG, netD, optimG, optimD, epoch, model_dir, saveD=False, saveOptim=False, max_to_keep=5):
    checkpoint = {
        'epoch': epoch,
        'netG': netG.state_dict(),
        'optimG': optimG.state_dict() if saveOptim else {},
        'netD': netD.state_dict() if saveD else {},
        'optimD': optimD.state_dict() if saveOptim else {}}
    torch.save(checkpoint, "{}/checkpoint_{:04}.pth".format(model_dir, epoch))
    print('Save G/D models')

    if max_to_keep is not None and max_to_keep > 0:
        checkpoint_list = sorted([ckpt for ckpt in glob.glob(model_dir + "/" + '*.pth')])
        while len(checkpoint_list) > max_to_keep:
            os.remove(checkpoint_list[0])
            checkpoint_list = checkpoint_list[1:]


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
