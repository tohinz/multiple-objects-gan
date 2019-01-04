from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import time
import numpy as np
import sys
import glob

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, resume):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.resume = resume

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]

        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

    def build_models(self):
        # ###################encoders######################################## #
        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = \
            RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = \
            torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        # #######################generator and discriminators############## #
        netsD = []
        from model import D_NET64, D_NET128, D_NET256
        netG = G_NET()
        if cfg.TREE.BRANCH_NUM > 0:
            netsD.append(D_NET64())
        if cfg.TREE.BRANCH_NUM > 1:
            netsD.append(D_NET128())
        if cfg.TREE.BRANCH_NUM > 2:
            netsD.append(D_NET256())

        netG.apply(weights_init)
        # print(netG)
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
            # print(netsD[i])
        print('# of netsD', len(netsD))
        epoch = 0

        if self.resume:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            latest_checkpoint = checkpoint_list[-1]
            state_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict["netG"])
            for i in range(len(netsD)):
                netsD[i].load_state_dict(state_dict["netD"][i])
            epoch = int(latest_checkpoint[-8:-4]) + 1
            print("Resuming training from checkpoint {} at epoch {}.".format(latest_checkpoint, epoch))

        #
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)
        # ########################################################### #
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()
        return [text_encoder, image_encoder, netG, netsD, epoch]

    def define_optimizers(self, netG, netsD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))

        if self.resume:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            latest_checkpoint = checkpoint_list[-1]
            state_dict = torch.load(latest_checkpoint, map_location=lambda storage, loc: storage)
            optimizerG.load_state_dict(state_dict["optimG"])

            for i in range(len(netsD)):
                optimizersD[i].load_state_dict(state_dict["optimD"][i])

        return optimizerG, optimizersD

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, optimG, optimsD, epoch, max_to_keep=5):
        netDs_state_dicts = []
        optimDs_state_dicts = []
        for i in range(len(netsD)):
            netD = netsD[i]
            optimD = optimsD[i]
            netDs_state_dicts.append(netD.state_dict())
            optimDs_state_dicts.append(optimD.state_dict())

        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        checkpoint = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'optimG': optimG.state_dict(),
            'netD': netDs_state_dicts,
            'optimD': optimDs_state_dicts}
        torch.save(checkpoint, "{}/checkpoint_{:04}.pth".format(self.model_dir, epoch))
        print('Save G/D models')

        load_params(netG, backup_para)

        if max_to_keep is not None and max_to_keep > 0:
            checkpoint_list = sorted([ckpt for ckpt in glob.glob(self.model_dir + "/" + '*.pth')])
            while len(checkpoint_list) > max_to_keep:
                os.remove(checkpoint_list[0])
                checkpoint_list = checkpoint_list[1:]

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results(self, netG, noise, sent_emb, words_embs, mask,
                         image_encoder, captions, cap_lens,
                         gen_iterations,  transf_matrices_inv, label_one_hot, name='current'):
        # Save images
        # fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
        inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
        fake_imgs, attention_maps, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
        for i in range(len(attention_maps)):
            if len(fake_imgs) > 1:
                img = fake_imgs[i + 1].detach().cpu()
                lr_img = fake_imgs[i].detach().cpu()
            else:
                img = fake_imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)

    def train(self):
        text_encoder, image_encoder, netG, netsD, start_epoch = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizersD = self.define_optimizers(netG, netsD)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            data_iter = iter(self.data_loader)
            step = 0
            while step < self.num_batches:
                # reset requires_grad to be trainable for all Ds
                # self.set_requires_grad_value(netsD, True)

                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                data = data_iter.next()
                imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot = prepare_data(data)
                transf_matrices = transformation_matrices[0]
                transf_matrices_inv = transformation_matrices[1]

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
                fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                #######################################################
                # (3) Update D network
                ######################################################
                errD_total = 0
                D_logs = ''
                for i in range(len(netsD)):
                    netsD[i].zero_grad()
                    if i == 0:
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  sent_emb, real_labels, fake_labels, self.gpus,
                                                  local_labels=label_one_hot, transf_matrices=transf_matrices,
                                                  transf_matrices_inv=transf_matrices_inv)
                    else:
                        errD = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                  sent_emb, real_labels, fake_labels, self.gpus)

                    # backward and update parameters
                    errD.backward()
                    optimizersD[i].step()
                    errD_total += errD
                    D_logs += 'errD%d: %.2f ' % (i, errD.item())

                #######################################################
                # (4) Update G network: maximize log(D(G(z)))
                ######################################################
                # compute total loss for training G
                step += 1
                gen_iterations += 1

                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total, G_logs = \
                    generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids, self.gpus,
                                   local_labels=label_one_hot, transf_matrices=transf_matrices,
                                   transf_matrices_inv=transf_matrices_inv)
                kl_loss = KL_loss(mu, logvar)
                errG_total += kl_loss
                G_logs += 'kl_loss: %.2f ' % kl_loss.item()
                # backward and update parameters
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # save images
                if gen_iterations % 1000 == 0:
                    print(D_logs + '\n' + G_logs)

                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results(netG, fixed_noise, sent_emb,
                                          words_embs, mask, image_encoder,
                                          captions, cap_lens, epoch,  transf_matrices_inv,
                                          label_one_hot, name='average')
                    load_params(netG, backup_para)
            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netsD, optimizerG, optimizersD, epoch)

        self.save_model(netG, avg_param_G, netsD, optimizerG, optimizersD, epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            # range from [-1, 1] to [0, 1]
            # img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir, num_samples=30000):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()
            #
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz))
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG.load_state_dict(state_dict["netG"])
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0

            for _ in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 10000 == 0:
                        print('step: ', step)
                    if step >= num_samples:
                        break

                    imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot = prepare_data(data)
                    transf_matrices_inv = transformation_matrices[1]

                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
                    mask = (captions == 0)
                    num_words = words_embs.size(2)
                    if mask.size(1) > num_words:
                        mask = mask[:, :num_words]

                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
                    with torch.no_grad():
                        fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    for j in range(batch_size):
                        s_tmp = '%s/single/%s' % (save_dir, keys[j])
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        # for k in range(len(fake_imgs)):
                        im = fake_imgs[k][j].data.cpu().numpy()
                        # [-1, 1] --> [0, 255]
                        im = (im + 1.0) * 127.5
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath = '%s_s%d.png' % (s_tmp, k)
                        im.save(fullpath)

    def sample(self, split_dir, num_samples=25, draw_bbox=False):
        from PIL import Image, ImageDraw, ImageFont
        import cPickle as pickle
        import torchvision
        import torchvision.utils as vutils

        if cfg.TRAIN.NET_G == '':
            print('Error: the path for model NET_G is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            # Build and load the generator
            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            batch_size = cfg.TRAIN.BATCH_SIZE
            nz = cfg.GAN.Z_DIM

            model_dir = cfg.TRAIN.NET_G
            state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            netG = G_NET()
            print('Load G from: ', model_dir)
            netG.apply(weights_init)

            netG.load_state_dict(state_dict["netG"])
            netG.cuda()
            netG.eval()

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s_%s' % (s_tmp, split_dir)
            mkdir_p(save_dir)
            #######################################
            noise = Variable(torch.FloatTensor(9, nz))

            imsize = 256

            for step, data in enumerate(self.data_loader, 0):
                if step >= num_samples:
                    break

                imgs, captions, cap_lens, class_ids, keys, transformation_matrices, label_one_hot, bbox = \
                    prepare_data(data, eval=True)
                transf_matrices_inv = transformation_matrices[1][0].unsqueeze(0)
                label_one_hot = label_one_hot[0].unsqueeze(0)

                img = imgs[-1][0]
                val_image = img.view(1, 3, imsize, imsize)
                val_image = (val_image - 0.5) * 2

                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs[0].unsqueeze(0).detach(), sent_emb[0].unsqueeze(0).detach()
                words_embs = words_embs.repeat(9, 1, 1)
                sent_emb = sent_emb.repeat(9, 1)
                mask = (captions == 0)
                mask = mask[0].unsqueeze(0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                mask = mask.repeat(9, 1)
                transf_matrices_inv = transf_matrices_inv.repeat(9, 1, 1, 1)
                label_one_hot = label_one_hot.repeat(9, 1, 1)

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (noise, sent_emb, words_embs, mask, transf_matrices_inv, label_one_hot)
                with torch.no_grad():
                    fake_imgs, _, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                data_img = torch.FloatTensor(10, 3, imsize, imsize).fill_(0)
                data_img[0] = val_image
                data_img[1:10] = fake_imgs[-1]

                if draw_bbox:
                    for idx in range(3):
                        x, y, w, h = tuple([int(imsize*x) for x in bbox[0, idx]])
                        w = imsize-1 if w > imsize-1 else w
                        h = imsize-1 if h > imsize-1 else h
                        if x <= -1:
                            break
                        data_img[:10, :, y, x:x + w] = 1
                        data_img[:10, :, y:y + h, x] = 1
                        data_img[:10, :, y+h, x:x + w] = 1
                        data_img[:10, :, y:y + h, x + w] = 1

                # get caption
                cap = captions[0].data.cpu().numpy()
                sentence = ""
                for j in range(len(cap)):
                    if cap[j] == 0:
                        break
                    word = self.ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
                    sentence += word + " "
                sentence = sentence[:-1]
                vutils.save_image(data_img, '{}/{}_{}.png'.format(save_dir, sentence, step), normalize=True, nrow=10)
            print("Saved {} files to {}".format(step, save_dir))

    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            # the path to save generated images
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size = captions.shape[0]
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1):  # 16
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()
                    #######################################################
                    # (1) Extract text embeddings
                    ######################################################
                    hidden = text_encoder.init_hidden(batch_size)
                    # words_embs: batch_size x nef x seq_len
                    # sent_emb: batch_size x nef
                    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                    mask = (captions == 0)
                    #######################################################
                    # (2) Generate fake images
                    ######################################################
                    noise.data.normal_(0, 1)
                    with torch.no_grad():
                        fake_imgs, attention_maps, _, _ = netG(noise, sent_emb, words_embs, mask)
                    # G attention
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            # print('im', im.shape)
                            im = np.transpose(im, (1, 2, 0))
                            # print('im', im.shape)
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)

                        for k in range(len(attention_maps)):
                            if len(fake_imgs) > 1:
                                im = fake_imgs[k + 1].detach().cpu()
                            else:
                                im = fake_imgs[0].detach().cpu()
                            attn_maps = attention_maps[k]
                            att_sze = attn_maps.size(2)
                            img_set, sentences = \
                                build_super_images2(im[j].unsqueeze(0),
                                                    captions[j].unsqueeze(0),
                                                    [cap_lens_np[j]], self.ixtoword,
                                                    [attn_maps[j]], att_sze)
                            if img_set is not None:
                                im = Image.fromarray(img_set)
                                fullpath = '%s_a%d.png' % (save_name, k)
                                im.save(fullpath)
