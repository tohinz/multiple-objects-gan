from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
import torchfile

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from miscc.utils import compute_transformation_matrix, compute_transformation_matrix_inverse
from miscc.utils import load_validation_data, pad_imgs

from tensorboard import summary
from tensorboard import FileWriter

class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.max_objects = 3

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    # ############# For training stageI GAN #############
    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        print(netG)
        netD = STAGE1_D()
        netD.apply(weights_init)
        print(netD)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict["netG"])
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,  map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD


    def train(self, data_loader):
        netG, netD = self.load_network_stageI()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))

        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1), requires_grad=False)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))

        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        print("Starting training...")
        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                real_img_cpu, bbox, label = data

                real_imgs = Variable(real_img_cpu)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    bbox = bbox.cuda()
                    label_one_hot = label.cuda().float()

                bbox = bbox.view(-1, 4)
                transf_matrices_inv = compute_transformation_matrix_inverse(bbox).float()
                transf_matrices_inv = transf_matrices_inv.view(real_imgs.shape[0], self.max_objects, 2, 3)
                transf_matrices = compute_transformation_matrix(bbox).float()
                transf_matrices = transf_matrices.view(real_imgs.shape[0], self.max_objects, 2, 3)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (noise, transf_matrices_inv, label_one_hot)
                # _, fake_imgs = nn.parallel.data_parallel(netG, inputs, self.gpus)
                _, fake_imgs = netG(noise, transf_matrices_inv, label_one_hot)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = \
                    compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                               real_labels, fake_labels,
                                               label_one_hot, transf_matrices, transf_matrices_inv, self.gpus)
                errD.backward(retain_graph=True)
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                errG = compute_generator_loss(netD, fake_imgs, real_labels, label_one_hot,
                                              transf_matrices, transf_matrices_inv, self.gpus)
                errG_total = errG
                errG_total.backward()
                optimizerG.step()

                ############################
                # (3) Log results
                ###########################
                count += 1
                if i % 500 == 0:
                    summary_D = summary.scalar('D_loss', errD.item())
                    summary_D_r = summary.scalar('D_loss_real', errD_real)
                    summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    summary_G = summary.scalar('G_loss', errG.item())

                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_D_r, count)
                    self.summary_writer.add_summary(summary_D_w, count)
                    self.summary_writer.add_summary(summary_D_f, count)
                    self.summary_writer.add_summary(summary_G, count)

                    # save the image result for each epoch
                    with torch.no_grad():
                        inputs = (noise, transf_matrices_inv, label_one_hot)
                        lr_fake, fake = nn.parallel.data_parallel(netG, inputs, self.gpus)
                        real_img_cpu = pad_imgs(real_img_cpu)
                        fake = pad_imgs(fake)
                        save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                        if lr_fake is not None:
                            save_img_results(None, lr_fake, epoch, self.image_dir)
            with torch.no_grad():
                inputs = (noise, transf_matrices_inv, label_one_hot)
                lr_fake, fake = nn.parallel.data_parallel(netG, inputs, self.gpus)
                real_img_cpu = pad_imgs(real_img_cpu)
                fake = pad_imgs(fake)
                save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                if lr_fake is not None:
                    save_img_results(None, lr_fake, epoch, self.image_dir)
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.item(), errG.item(),
                     errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, optimizerG, optimizerD, epoch, self.model_dir)
        #
        save_model(netG, netD, optimizerG, optimizerD, epoch, self.model_dir)
        #
        self.summary_writer.close()

    def sample(self, datapath, num_samples=25, draw_bbox=True, num_digits_per_img=3, change_bbox_size=False):
        from PIL import Image, ImageDraw, ImageFont
        import cPickle as pickle
        import torchvision
        import torchvision.utils as vutils
        img_dir = os.path.join(datapath, "normal", "imgs/")
        netG, _ = self.load_network_stageI()
        netG.eval()
        test_set_size = 10000

        label, bbox = load_validation_data(datapath)
        if num_digits_per_img < 3:
            label = label[:, :num_digits_per_img, :]
            bbox = bbox[:, :num_digits_per_img, ...]
        elif num_digits_per_img > 3:
            def get_one_hot(targets, nb_classes):
                res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
                return res.reshape(list(targets.shape) + [nb_classes])

            labels_sample = np.random.randint(0, 10, size=(bbox.shape[0], num_digits_per_img-3))
            labels_sample = get_one_hot(labels_sample, 10)
            labels_new = np.zeros((label.shape[0], num_digits_per_img, 10))
            labels_new[:, :3, :] = label
            labels_new[:, 3:, :] = labels_sample
            label = torch.from_numpy(labels_new)

            bboxes_x = np.random.random((bbox.shape[0], num_digits_per_img-3, 1))
            bboxes_y = np.random.random((bbox.shape[0], num_digits_per_img-3, 1))
            bboxes_w = np.random.randint(10, 20, size=(bbox.shape[0], num_digits_per_img-3, 1)) / 64.0
            bboxes_h = np.random.randint(16, 20, size=(bbox.shape[0], num_digits_per_img-3, 1)) / 64.0

            bbox_new_concat = np.concatenate((bboxes_x, bboxes_y, bboxes_w, bboxes_h), axis=2)
            bbox_new = np.zeros([bbox.shape[0], num_digits_per_img, 4])
            bbox_new[:, :3, :] = bbox
            bbox_new[:, 3:, :] = bbox_new_concat
            bbox = torch.from_numpy(bbox_new)

        if change_bbox_size:
            bbox_idx = np.random.randint(0, bbox.shape[1])
            scale_x = np.random.random(bbox.shape[0])
            scale_x[scale_x < 0.5] = 0.5
            scale_y = np.random.random(bbox.shape[0])
            scale_y[scale_y < 0.5] = 0.5

            bbox[:, bbox_idx, 2] *= torch.from_numpy(scale_x)
            bbox[:, bbox_idx, 3] *= torch.from_numpy(scale_y)

        filepath = os.path.join(datapath, "normal", 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')] + "_samples_" + str(num_digits_per_img) + "_digits"
        if change_bbox_size:
            save_dir += "_change_bbox_size"
        print("Saving {} to {}:".format(num_samples, save_dir))
        mkdir_p(save_dir)
        if cfg.CUDA:
            bbox = bbox.cuda()
            label_one_hot = label.cuda().float()

        #######################################
        bbox_ = bbox.clone()
        bbox = bbox.view(-1, 4)
        transf_matrices_inv = compute_transformation_matrix_inverse(bbox).float()
        transf_matrices_inv = transf_matrices_inv.view(test_set_size, num_digits_per_img, 2, 3)
        #######################################

        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(9, nz))
        if cfg.CUDA:
            noise = noise.cuda()

        imsize = 64

        for count in range(num_samples):
            index = int(np.random.randint(0, test_set_size, 1))
            key = filenames[index].split("/")[-1]
            img_name = img_dir + key
            img = Image.open(img_name)
            val_image = torchvision.transforms.functional.to_tensor(img)
            val_image = val_image.view(1, 1, imsize, imsize)
            val_image = (val_image - 0.5) * 2

            transf_matrices_inv_batch = transf_matrices_inv[index]
            label_one_hot_batch = label_one_hot[index]

            transf_matrices_inv_batch = transf_matrices_inv_batch.view(1, num_digits_per_img, 2, 3).repeat(9, 1, 1, 1)
            label_one_hot_batch = label_one_hot_batch.view(1, num_digits_per_img, 10).repeat(9, 1, 1)

            if cfg.CUDA:
                label_one_hot_batch = label_one_hot_batch.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (noise, transf_matrices_inv_batch, label_one_hot_batch, num_digits_per_img)
            _, fake_imgs = nn.parallel.data_parallel(netG, inputs, self.gpus)

            data_img = torch.FloatTensor(20, 1, imsize, imsize).fill_(0)
            data_img[0] = val_image
            data_img[1:10] = fake_imgs

            if draw_bbox:
                for idx in range(num_digits_per_img):
                    x, y, w, h = tuple([int(imsize*x) for x in bbox_[index, idx]])
                    w = imsize-1 if w > imsize-1 else w
                    h = imsize-1 if h > imsize-1 else h
                    while x + w >= 64:
                        x -= 1
                        w -= 1
                    while y + h >= 64:
                        y -= 1
                        h -= 1
                    if x <= -1:
                        break
                    data_img[:10, :, y, x:x + w] = 1
                    data_img[:10, :, y:y + h, x] = 1
                    data_img[:10, :, y+h, x:x + w] = 1
                    data_img[:10, :, y:y + h, x + w] = 1

            # write digit identities into image
            text_img = Image.new('L', (imsize*10, imsize), color = 'white')
            d = ImageDraw.Draw(text_img)
            label = label_one_hot_batch[0]
            label = label.cpu().numpy()
            label = np.argmax(label, axis=1)
            label = ", ".join([str(label[_]) for _ in range(num_digits_per_img)])
            d.text((10,10), label)
            text_img = torchvision.transforms.functional.to_tensor(text_img)
            text_img = torch.chunk(text_img, 10, 2)
            text_img = torch.cat([text_img[i].view(1, 1, imsize, imsize) for i in range(10)], 0)
            data_img[10:] = text_img
            vutils.save_image(data_img, '{}/vis_{}.png'.format(save_dir, count), normalize=True, nrow=10)
        print("Saved {} files to {}".format(count+1, save_dir))

