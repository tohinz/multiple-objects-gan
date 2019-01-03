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
from miscc.utils import load_validation_data

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
        self.max_objects = 4

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
        netD = STAGE1_D()
        netD.apply(weights_init)

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

    def train(self, data_loader, stage=1):
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

        print("Start training...")
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
                real_img_cpu, transformation_matrices, label_one_hot, _ = data

                transf_matrices, transf_matrices_inv = tuple(transformation_matrices)
                transf_matrices = transf_matrices.detach()
                transf_matrices_inv = transf_matrices_inv.detach()

                real_imgs = Variable(real_img_cpu)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    label_one_hot = label_one_hot.cuda()
                    transf_matrices = transf_matrices.cuda()
                    transf_matrices_inv = transf_matrices_inv.cuda()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (noise, transf_matrices_inv, label_one_hot)
                fake_imgs = nn.parallel.data_parallel(netG, inputs, self.gpus)

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
                count = count + 1
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
                        fake = nn.parallel.data_parallel(netG, inputs, self.gpus)
                        save_img_results(real_img_cpu, fake, epoch, self.image_dir)
            with torch.no_grad():
                inputs = (noise, transf_matrices_inv, label_one_hot)
                fake = nn.parallel.data_parallel(netG, inputs, self.gpus)
                save_img_results(real_img_cpu, fake, epoch, self.image_dir)
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


    def sample(self, data_loader, num_samples=25, draw_bbox=True, max_objects=4):
        from PIL import Image, ImageDraw, ImageFont
        import cPickle as pickle
        import torchvision
        import torchvision.utils as vutils
        netG, _ = self.load_network_stageI()
        netG.eval()

        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')] + "_samples_" + str(max_objects) + "_objects"
        print("saving to:", save_dir)
        mkdir_p(save_dir)

        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(9, nz))
        if cfg.CUDA:
            noise = noise.cuda()

        imsize = 64

        # for count in range(num_samples):
        count = 0
        for i, data in enumerate(data_loader, 0):
            if count == num_samples:
                break
            ######################################################
            # (1) Prepare training data
            ######################################################
            real_img_cpu, transformation_matrices, label_one_hot, bbox = data

            transf_matrices, transf_matrices_inv = tuple(transformation_matrices)
            transf_matrices_inv = transf_matrices_inv.detach()

            real_img = Variable(real_img_cpu)
            if cfg.CUDA:
                real_img = real_img.cuda()
                label_one_hot = label_one_hot.cuda()
                transf_matrices_inv = transf_matrices_inv.cuda()

            transf_matrices_inv_batch = transf_matrices_inv.view(1, max_objects, 2, 3).repeat(9, 1, 1, 1)
            label_one_hot_batch = label_one_hot.view(1, max_objects, 13).repeat(9, 1, 1)

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (noise, transf_matrices_inv_batch, label_one_hot_batch)
            with torch.no_grad():
                fake_imgs= nn.parallel.data_parallel(netG, inputs, self.gpus)

            data_img = torch.FloatTensor(20, 3, imsize, imsize).fill_(0)
            data_img[0] = real_img
            data_img[1:10] = fake_imgs

            if draw_bbox:
                for idx in range(max_objects):
                    x, y, w, h = tuple([int(imsize*x) for x in bbox[0, idx]])
                    w = imsize-1 if w > imsize-1 else w
                    h = imsize-1 if h > imsize-1 else h
                    if x <= -1 or y <= -1:
                        break
                    data_img[:10, :, y, x:x + w] = 1
                    data_img[:10, :, y:y + h, x] = 1
                    data_img[:10, :, y+h, x:x + w] = 1
                    data_img[:10, :, y:y + h, x + w] = 1

            # write caption into image
            shape_dict = {
                0: "cube",
                1: "cylinder",
                2: "sphere",
                3: "empty"
            }

            color_dict = {
                0: "gray",
                1: "red",
                2: "blue",
                3: "green",
                4: "brown",
                5: "purple",
                6: "cyan",
                7: "yellow",
                8: "empty"
            }
            text_img = Image.new('L', (imsize * 10, imsize), color='white')
            d = ImageDraw.Draw(text_img)
            label = label_one_hot_batch[0]
            label = label.cpu().numpy()
            label_shape = label[:, :4]
            label_color = label[:, 4:]
            label_shape = np.argmax(label_shape, axis=1)
            label_color = np.argmax(label_color, axis=1)
            label_combined = ", ".join([color_dict[label_color[_]] + " " + shape_dict[label_shape[_]]
                                        for _ in range(max_objects)])
            d.text((10, 10), label_combined)
            text_img = torchvision.transforms.functional.to_tensor(text_img)
            text_img = torch.chunk(text_img, 10, 2)
            text_img = torch.cat([text_img[i].view(1, 1, imsize, imsize) for i in range(10)], 0)
            data_img[10:] = text_img
            vutils.save_image(data_img, '{}/vis_{}.png'.format(save_dir, count), normalize=True, nrow=10)
            count += 1

        print("Saved {} files to {}".format(count, save_dir))
