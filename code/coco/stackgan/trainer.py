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
from miscc.utils import KL_loss
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

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G(Stage1_G)
        netG.apply(weights_init)
        print(netG)
        if cfg.NET_G != '':
            state_dict = torch.load(cfg.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict["netG"])
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = torch.load(cfg.STAGE1_G, map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict["netG"])
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1, max_objects=3):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        # with torch.no_grad():
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
                real_img_cpu, bbox, label, txt_embedding = data

                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    if cfg.STAGE == 1:
                        bbox = bbox.cuda()
                    elif cfg.STAGE == 2:
                        bbox = [bbox[0].cuda(), bbox[1].cuda()]
                    label = label.cuda()
                    txt_embedding = txt_embedding.cuda()

                if cfg.STAGE == 1:
                    bbox = bbox.view(-1, 4)
                    transf_matrices_inv = compute_transformation_matrix_inverse(bbox)
                    transf_matrices_inv = transf_matrices_inv.view(real_imgs.shape[0], max_objects, 2, 3)
                    transf_matrices = compute_transformation_matrix(bbox)
                    transf_matrices = transf_matrices.view(real_imgs.shape[0], max_objects, 2, 3)
                elif cfg.STAGE == 2:
                    _bbox = bbox[0].view(-1, 4)
                    transf_matrices_inv = compute_transformation_matrix_inverse(_bbox)
                    transf_matrices_inv = transf_matrices_inv.view(real_imgs.shape[0], max_objects, 2, 3)

                    _bbox = bbox[1].view(-1, 4)
                    transf_matrices_inv_s2 = compute_transformation_matrix_inverse(_bbox)
                    transf_matrices_inv_s2 = transf_matrices_inv_s2.view(real_imgs.shape[0], max_objects, 2, 3)
                    transf_matrices_s2 = compute_transformation_matrix(_bbox)
                    transf_matrices_s2 = transf_matrices_s2.view(real_imgs.shape[0], max_objects, 2, 3)

                # produce one-hot encodings of the labels
                _labels = label.long()
                # remove -1 to enable one-hot converting
                _labels[_labels < 0] = 80
                label_one_hot = torch.cuda.FloatTensor(noise.shape[0], max_objects, 81).fill_(0)
                label_one_hot = label_one_hot.scatter_(2, _labels, 1).float()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                if cfg.STAGE == 1:
                    inputs = (txt_embedding, noise, transf_matrices_inv, label_one_hot)
                elif cfg.STAGE == 2:
                    inputs = (txt_embedding, noise, transf_matrices_inv, transf_matrices_s2, transf_matrices_inv_s2, label_one_hot)
                _, fake_imgs, mu, logvar, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
                # _, fake_imgs, mu, logvar, _ = netG(txt_embedding, noise, transf_matrices_inv, label_one_hot)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()

                if cfg.STAGE == 1:
                    errD, errD_real, errD_wrong, errD_fake = \
                        compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                                   real_labels, fake_labels,
                                                   label_one_hot, transf_matrices, transf_matrices_inv,
                                                   mu, self.gpus)
                elif cfg.STAGE == 2:
                    errD, errD_real, errD_wrong, errD_fake = \
                        compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                                   real_labels, fake_labels,
                                                   label_one_hot, transf_matrices_s2, transf_matrices_inv_s2,
                                                   mu, self.gpus)
                errD.backward(retain_graph=True)
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                netG.zero_grad()
                if cfg.STAGE == 1:
                    errG = compute_generator_loss(netD, fake_imgs,
                                                  real_labels, label_one_hot, transf_matrices, transf_matrices_inv,
                                                  mu, self.gpus)
                elif cfg.STAGE == 2:
                    errG = compute_generator_loss(netD, fake_imgs,
                                                  real_labels, label_one_hot, transf_matrices_s2, transf_matrices_inv_s2,
                                                  mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                errG_total.backward()
                optimizerG.step()

                count += 1
                if i % 500 == 0:
                    summary_D = summary.scalar('D_loss', errD.item())
                    summary_D_r = summary.scalar('D_loss_real', errD_real)
                    summary_D_w = summary.scalar('D_loss_wrong', errD_wrong)
                    summary_D_f = summary.scalar('D_loss_fake', errD_fake)
                    summary_G = summary.scalar('G_loss', errG.item())
                    summary_KL = summary.scalar('KL_loss', kl_loss.item())

                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_D_r, count)
                    self.summary_writer.add_summary(summary_D_w, count)
                    self.summary_writer.add_summary(summary_D_f, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                    # save the image result for each epoch
                    with torch.no_grad():
                        if cfg.STAGE == 1:
                            inputs = (txt_embedding, noise, transf_matrices_inv, label_one_hot)
                        elif cfg.STAGE == 2:
                            inputs = (txt_embedding, noise, transf_matrices_inv, transf_matrices_s2, transf_matrices_inv_s2, label_one_hot)
                        lr_fake, fake, _, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
                        save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                        if lr_fake is not None:
                            save_img_results(None, lr_fake, epoch, self.image_dir)
            with torch.no_grad():
                if cfg.STAGE == 1:
                    inputs = (txt_embedding, noise, transf_matrices_inv, label_one_hot)
                elif cfg.STAGE == 2:
                    inputs = (txt_embedding, noise, transf_matrices_inv, transf_matrices_s2, transf_matrices_inv_s2, label_one_hot)
                lr_fake, fake, _, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
                save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                if lr_fake is not None:
                    save_img_results(None, lr_fake, epoch, self.image_dir)
            end_t = time.time()
            print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                     Total Time: %.2fsec
                  '''
                  % (epoch, self.max_epoch, i, len(data_loader),
                     errD.item(), errG.item(), kl_loss.item(),
                     errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, optimizerG, optimizerD, epoch, self.model_dir)
        #
        save_model(netG, netD, optimizerG, optimizerD, epoch, self.model_dir)
        #
        self.summary_writer.close()

    def sample(self, datapath, num_samples=25, stage=1, draw_bbox=True, max_objects=3):
        from PIL import Image, ImageDraw, ImageFont
        import cPickle as pickle
        import torchvision
        import torchvision.utils as vutils
        img_dir = cfg.IMG_DIR
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath + "val_captions.t7")
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        label, bbox = load_validation_data(datapath)

        filepath = os.path.join(datapath, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')] + "_visualize_bbox"
        print("saving to:", save_dir)
        mkdir_p(save_dir)

        if cfg.CUDA:
            if cfg.STAGE == 1:
                bbox = bbox.cuda()
            elif cfg.STAGE == 2:
                bbox = [bbox.clone().cuda(), bbox.cuda()]
            label = label.cuda()

        #######################################
        if cfg.STAGE == 1:
            bbox_ = bbox.clone()
        elif cfg.STAGE == 2:
            bbox_ = bbox[0].clone()

        if cfg.STAGE == 1:
            bbox = bbox.view(-1, 4)
            transf_matrices_inv = compute_transformation_matrix_inverse(bbox)
            transf_matrices_inv = transf_matrices_inv.view(num_embeddings, max_objects, 2, 3)
        elif cfg.STAGE == 2:
            _bbox = bbox[0].view(-1, 4)
            transf_matrices_inv = compute_transformation_matrix_inverse(_bbox)
            transf_matrices_inv = transf_matrices_inv.view(num_embeddings, max_objects, 2, 3)

            _bbox = bbox[1].view(-1, 4)
            transf_matrices_inv_s2 = compute_transformation_matrix_inverse(_bbox)
            transf_matrices_inv_s2 = transf_matrices_inv_s2.view(num_embeddings, max_objects, 2, 3)
            transf_matrices_s2 = compute_transformation_matrix(_bbox)
            transf_matrices_s2 = transf_matrices_s2.view(num_embeddings, max_objects, 2, 3)

        # produce one-hot encodings of the labels
        _labels = label.long()
        # remove -1 to enable one-hot converting
        _labels[_labels < 0] = 80
        label_one_hot = torch.cuda.FloatTensor(num_embeddings, max_objects, 81).fill_(0)
        label_one_hot = label_one_hot.scatter_(2, _labels, 1).float()
        #######################################

        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(9, nz))
        if cfg.CUDA:
            noise = noise.cuda()

        imsize = 64 if stage == 1 else 256

        for count in range(num_samples):
            index = int(np.random.randint(0, num_embeddings, 1))
            key = filenames[index]
            img_name = img_dir + "/" + key + ".jpg"
            img = Image.open(img_name).convert('RGB').resize((imsize, imsize), Image.ANTIALIAS)
            val_image = torchvision.transforms.functional.to_tensor(img)
            val_image = val_image.view(1, 3, imsize, imsize)
            val_image = (val_image - 0.5) * 2

            embeddings_batch = embeddings[index]
            transf_matrices_inv_batch = transf_matrices_inv[index]
            label_one_hot_batch = label_one_hot[index]

            embeddings_batch = np.reshape(embeddings_batch, (1, 1024)).repeat(9,0)
            transf_matrices_inv_batch = transf_matrices_inv_batch.view(1, 3, 2, 3).repeat(9, 1, 1, 1)
            label_one_hot_batch = label_one_hot_batch.view(1, 3, 81).repeat(9, 1, 1)

            if cfg.STAGE == 2:
                transf_matrices_s2_batch = transf_matrices_s2[index]
                transf_matrices_s2_batch = transf_matrices_s2_batch.view(1, 3, 2, 3).repeat(9, 1, 1, 1)
                transf_matrices_inv_s2_batch = transf_matrices_inv_s2[index]
                transf_matrices_inv_s2_batch = transf_matrices_inv_s2_batch.view(1, 3, 2, 3).repeat(9, 1, 1, 1)

            txt_embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                label_one_hot_batch = label_one_hot_batch.cuda()
                txt_embedding = txt_embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            # inputs = (txt_embedding, noise, transf_matrices_inv_batch, label_one_hot_batch)
            if cfg.STAGE == 1:
                inputs = (txt_embedding, noise, transf_matrices_inv_batch, label_one_hot_batch)
            elif cfg.STAGE == 2:
                inputs = (txt_embedding, noise, transf_matrices_inv_batch,
                          transf_matrices_s2_batch, transf_matrices_inv_s2_batch, label_one_hot_batch)
            _, fake_imgs, mu, logvar, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)

            data_img = torch.FloatTensor(10, 3, imsize, imsize).fill_(0)
            data_img[0] = val_image
            data_img[1:10] = fake_imgs

            if draw_bbox:
                for idx in range(3):
                    x, y, w, h = tuple([int(imsize*x) for x in bbox_[index, idx]])
                    w = imsize-1 if w > imsize-1 else w
                    h = imsize-1 if h > imsize-1 else h
                    if x <= -1:
                        break
                    data_img[:10, :, y, x:x + w] = 1
                    data_img[:10, :, y:y + h, x] = 1
                    data_img[:10, :, y+h, x:x + w] = 1
                    data_img[:10, :, y:y + h, x + w] = 1

            vutils.save_image(data_img, '{}/{}.png'.format(save_dir, captions_list[index]), normalize=True, nrow=10)

        print("Saved {} files to {}".format(count+1, save_dir))

