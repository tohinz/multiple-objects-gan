import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from miscc.utils import compute_transformation_matrix, compute_transformation_matrix_inverse
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))#,
                # nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))#,
                # nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


def stn(image, transformation_matrix, size):
    grid = torch.nn.functional.affine_grid(transformation_matrix, torch.Size(size))
    out_image = torch.nn.functional.grid_sample(image, grid)

    return out_image


class BBOX_NET(nn.Module):
    def __init__(self):
        super(BBOX_NET, self).__init__()
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.encode = nn.Sequential(
            # 128 * 16 x 16
            conv3x3(self.c_dim, self.c_dim // 2, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 8 x 8
            conv3x3(self.c_dim // 2, self.c_dim // 4, stride=2),
            nn.BatchNorm2d(self.c_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 4 x 4
            conv3x3(self.c_dim // 4, self.c_dim // 8, stride=2),
            nn.BatchNorm2d(self.c_dim // 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 2 x 2
         )

    def forward(self, labels, transf_matr_inv, max_objects):
        label_layout = torch.cuda.FloatTensor(labels.shape[0], self.c_dim, 16, 16).fill_(0)
        for idx in range(max_objects):
            current_label = labels[:, idx]
            current_label = current_label.view(current_label.shape[0], current_label.shape[1], 1, 1)
            current_label = current_label.repeat(1, 1, 16, 16)
            current_label = stn(current_label, transf_matr_inv[:, idx], current_label.shape)
            label_layout += current_label

        layout_encoding = self.encode(label_layout).view(labels.shape[0], -1)

        return layout_encoding

# ############# Networks for stageI GAN #############
class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        linput = self.ef_dim + 81
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.USE_BBOX_LAYOUT:
            self.bbox_net = BBOX_NET()
            ninput += 64

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # local pathway
        self.label = nn.Sequential(
            nn.Linear(linput, self.ef_dim, bias=False),
            nn.BatchNorm1d(self.ef_dim),
            nn.ReLU(True))
        self.local1 = upBlock(self.ef_dim, ngf // 2)
        self.local2 = upBlock(ngf // 2, ngf // 4)

        # global pathway
        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 2, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise, transf_matrices_inv, label_one_hot, max_objects=3):
        c_code, mu, logvar = self.ca_net(text_embedding)
        local_labels = torch.cuda.FloatTensor(noise.shape[0], max_objects, self.ef_dim).fill_(0)

        # local, object pathway
        # h_code_locals is the empty canvas on which the features are added at the locations given by the bbox
        h_code_locals = torch.cuda.FloatTensor(noise.shape[0], self.gf_dim // 4, 16, 16).fill_(0)
        for idx in range(max_objects):
            # generate individual label for each bounding box, based on bbox label and caption
            current_label = self.label(torch.cat((c_code, label_one_hot[:, idx]), 1))
            local_labels[:, idx] = current_label
            # replicate label spatially
            current_label = current_label.view(current_label.shape[0], self.ef_dim, 1, 1)
            current_label = current_label.repeat(1, 1, 4, 4)
            # apply object pathway to the label to generate object features
            h_code_local = self.local1(current_label)
            h_code_local = self.local2(h_code_local)
            # transform features to the shape of the bounding box and add to empty canvas
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], h_code_local.shape)
            h_code_locals += h_code_local

        # global pathway
        if cfg.USE_BBOX_LAYOUT:
            bbox_code = self.bbox_net(local_labels, transf_matrices_inv, max_objects)
            z_c_code = torch.cat((noise, c_code, bbox_code), 1)
        else:
            z_c_code = torch.cat((noise, c_code), 1)
        # start global pathway
        h_code = self.fc(z_c_code)
        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)

        # combine local and global
        h_code = torch.cat((h_code, h_code_locals), 1)

        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar, local_labels


class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim

        # local pathway
        self.local = nn.Sequential(
            nn.Conv2d(3 + 81, ndf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf*4, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def _encode_img(self, image, label, transf_matrices, transf_matrices_inv, max_objects):
        # local pathway
        # h_code_locals is the empty canvas on which the features are added at the locations given by the bbox
        h_code_locals = torch.cuda.FloatTensor(image.shape[0], self.df_dim * 2, 16, 16).fill_(0)
        for idx in range(max_objects):
            # get bbox label and replicate spatially
            current_label = label[:, idx].view(label.shape[0], 81, 1, 1)
            current_label = current_label.repeat(1, 1, 16, 16)
            # extract features from bounding box and concatenate with the bbox label
            h_code_local = stn(image, transf_matrices[:, idx], (image.shape[0], image.shape[1], 16, 16))
            h_code_local = torch.cat((h_code_local, current_label), 1)
            # apply local pathway
            h_code_local = self.local(h_code_local)
            # reshape extracted features to bbox layout and add to empty canvas
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], (h_code_local.shape[0], h_code_local.shape[1], 16, 16))
            h_code_locals += h_code_local

        # start global pathway
        h_code = self.conv1(image)
        h_code = self.act(h_code)
        h_code = self.conv2(h_code)
        h_code = self.bn2(h_code)
        h_code = self.act(h_code)

        # combine global and local pathway
        h_code = torch.cat((h_code, h_code_locals), 1)

        h_code = self.conv3(h_code)
        h_code = self.bn3(h_code)
        h_code = self.act(h_code)

        h_code = self.conv4(h_code)
        h_code = self.bn4(h_code)
        h_code = self.act(h_code)
        return h_code

    def forward(self, image, label, transf_matrices, transf_matrices_inv, max_objects=3):  # , label_one_hot):
        img_embedding = self._encode_img(image, label, transf_matrices, transf_matrices_inv, max_objects)

        return img_embedding


# ############# Networks for stageII GAN #############
class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # local pathway
        linput = self.ef_dim + 81
        self.label = nn.Sequential(
            nn.Linear(linput, self.ef_dim, bias=False),
            nn.BatchNorm1d(self.ef_dim),
            nn.ReLU(True))
        self.local1 = upBlock(self.ef_dim+768, ngf * 2)
        self.local2 = upBlock(ngf * 2, ngf)

        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        if cfg.USE_BBOX_LAYOUT:
            self.hr_joint = nn.Sequential(
                conv3x3(self.ef_dim * 2 + ngf * 4, ngf * 4),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True))
        else:
            self.hr_joint = nn.Sequential(
                conv3x3(self.ef_dim + ngf * 4, ngf * 4),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 32 x 32
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock(ngf * 2, ngf // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 256 x 256
        self.img = nn.Sequential(
            conv3x3(ngf // 4, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise, transf_matrices_inv,
                transf_matrices_s2, transf_matrices_inv_s2, label_one_hot, max_objects=3):
        _, stage1_img, _, _, _ = self.STAGE1_G(text_embedding, noise, transf_matrices_inv, label_one_hot)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        # contains the generated labels of the individual bboxes
        local_labels = torch.cuda.FloatTensor(noise.shape[0], max_objects, self.ef_dim).fill_(0)

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code_ = c_code.view(-1, self.ef_dim, 1, 1)
        c_code_ = c_code_.repeat(1, 1, 16, 16)

        if cfg.USE_BBOX_LAYOUT:
            labels_layout = torch.cuda.FloatTensor(noise.shape[0], self.ef_dim, 16, 16).fill_(0)
            # create bbox layout by adding the bbox labels at the locations of the bbox, zeros everywhere else
            for idx in range(max_objects):
                # first, generate labels for each bbox, using the one-hot bbox labels and image caption
                current_label = self.label(torch.cat((c_code, label_one_hot[:, idx]), 1))
                local_labels[:, idx] = current_label
                # replicate label spatially
                current_label = current_label.view(current_label.shape[0], current_label.shape[1], 1, 1)
                current_label = current_label.repeat(1, 1, 16, 16)
                # transfer label to bbox location and add to empty canvas
                label_local = stn(current_label, transf_matrices_inv[:, idx],
                                  (labels_layout.shape[0], labels_layout.shape[1], 16, 16))
                labels_layout += label_local
            # concatenate with the other information
            i_c_code = torch.cat([encoded_img, c_code_, labels_layout], 1)
        else:
            i_c_code = torch.cat([encoded_img, c_code_], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        # local pathway
        h_code_locals = torch.cuda.FloatTensor(h_code.shape[0], self.gf_dim, 64, 64).fill_(0)
        for idx in range(max_objects):
            if not cfg.USE_BBOX_LAYOUT:
                # generate local labels if not already done
                current_label = self.label(torch.cat((c_code, label_one_hot[:, idx]), 1))
                local_labels[:, idx] = current_label
            # replicate local labels spatially
            current_label = local_labels[:, idx].view(h_code.shape[0], 128, 1, 1)
            current_label = current_label.repeat(1, 1, 16, 16)
            # extract features from image at the location of the bbox and concat with label
            current_patch = stn(h_code, transf_matrices_s2[:, idx], (h_code.shape[0], h_code.shape[1], 16, 16))
            current_input = torch.cat((current_patch, current_label), 1)
            # apply local pathway
            h_code_local = self.local1(current_input)
            h_code_local = self.local2(h_code_local)
            # transfer features to bbox location and add to empty canvas
            h_code_local = stn(h_code_local, transf_matrices_inv_s2[:, idx], h_code_locals.shape)
            h_code_locals += h_code_local

        # start upsampling with global pathway
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)

        # combine global and local
        h_code = torch.cat((h_code, h_code_locals), 1)

        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar, local_labels


class STAGE2_D(nn.Module):
    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim

        self.local = nn.Sequential(
            nn.Conv2d(3 + 81, ndf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 6, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(ndf * 16)
        self.conv6 = nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(ndf * 32)
        self.conv7 = conv3x3(ndf * 32, ndf * 16)
        self.bn7 = nn.BatchNorm2d(ndf * 16)
        self.conv8 = conv3x3(ndf * 16, ndf * 8)
        self.bn8 = nn.BatchNorm2d(ndf * 8)


        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def _encode_img(self, image, label, transf_matrices, transf_matrices_inv, max_objects):
        # local pathway
        h_code_locals = torch.cuda.FloatTensor(image.shape[0], self.df_dim * 2, 32, 32).fill_(0)
        for idx in range(max_objects):
            # get current bbox label and replicate spatially
            current_label = label[:, idx]
            current_label = current_label.view(label.shape[0], 81, 1, 1)
            current_label = current_label.repeat(1, 1, 32, 32)
            # extract features from bbox and concat with label
            h_code_local = stn(image, transf_matrices[:, idx], (image.shape[0], image.shape[1], 32, 32))
            h_code_local = torch.cat((h_code_local, current_label), 1)
            # apply local pathway
            h_code_local = self.local(h_code_local)
            # transfer features to location of bbox and add to empty canvas
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], h_code_locals.shape)
            h_code_locals += h_code_local

        # start downsampling with global pathway
        h_code = self.conv1(image)
        h_code = self.act(h_code)
        h_code = self.conv2(h_code)
        h_code = self.bn2(h_code)
        h_code = self.act(h_code)
        h_code = self.conv3(h_code)
        h_code = self.bn3(h_code)
        h_code = self.act(h_code)

        # combine global and local
        h_code = torch.cat((h_code, h_code_locals), 1)

        h_code = self.conv4(h_code)
        h_code = self.bn4(h_code)
        h_code = self.act(h_code)
        h_code = self.conv5(h_code)
        h_code = self.bn5(h_code)
        h_code = self.act(h_code)
        h_code = self.conv6(h_code)
        h_code = self.bn6(h_code)
        h_code = self.act(h_code)
        h_code = self.conv7(h_code)
        h_code = self.bn7(h_code)
        h_code = self.act(h_code)
        h_code = self.conv8(h_code)
        h_code = self.bn8(h_code)
        h_code = self.act(h_code)

        return h_code

    def forward(self, image, label, transf_matrices, transf_matrices_inv, max_objects=3):
        img_embedding = self._encode_img(image, label, transf_matrices, transf_matrices_inv, max_objects)

        return img_embedding
