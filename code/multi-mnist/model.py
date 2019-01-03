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
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(c_code.shape[0], 10, 1, 1)
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
        self.c_dim = 128
        self.encode = nn.Sequential(
            # 128 * 16 x 16
            conv3x3(10, self.c_dim // 2, stride=2),
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

    def forward(self, labels, transf_matr_inv, num_digits):
        label_layout = torch.cuda.FloatTensor(labels.shape[0], 10, 16, 16).fill_(0)
        for idx in range(num_digits):
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
        self.ef_dim = 10
        self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        ninput = self.z_dim
        linput = self.ef_dim
        ngf = self.gf_dim

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
            conv3x3(ngf // 16, 1),
            nn.Tanh())

    def forward(self, noise, transf_matrices_inv, label_one_hot, num_digits_per_image=3):
        # local pathway
        h_code_locals = torch.cuda.FloatTensor(noise.shape[0], self.gf_dim // 4, 16, 16).fill_(0)

        for idx in range(num_digits_per_image):
            current_label = label_one_hot[:, idx]
            current_label = current_label.view(current_label.shape[0], self.ef_dim, 1, 1)
            current_label = current_label.repeat(1, 1, 4, 4)
            h_code_local = self.local1(current_label)
            h_code_local = self.local2(h_code_local)
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], h_code_local.shape)
            h_code_locals += h_code_local

        # global pathway
        if cfg.USE_BBOX_LAYOUT:
            bbox_code = self.bbox_net(label_one_hot, transf_matrices_inv, num_digits_per_image)
            z_c_code = torch.cat((noise, bbox_code), 1)
        else:
            z_c_code = torch.cat((noise), 1)
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
        return None, fake_img


class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = 10
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim

        # local pathway
        self.local = nn.Sequential(
            nn.Conv2d(1 + 10, ndf * 2, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(1, ndf, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf*4, ndf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf * 8)

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def _encode_img(self, image, label, transf_matrices, transf_matrices_inv, num_digits_per_image=3):
        # local pathway
        h_code_locals = torch.cuda.FloatTensor(image.shape[0], self.df_dim * 2, 16, 16).fill_(0)

        for idx in range(num_digits_per_image):
            current_label = label[:, idx].view(label.shape[0], 10, 1, 1)
            current_label = current_label.repeat(1, 1, 16, 16)
            h_code_local = stn(image, transf_matrices[:, idx], (image.shape[0], image.shape[1], 16, 16))
            h_code_local = torch.cat((h_code_local, current_label), 1)
            h_code_local = self.local(h_code_local)
            h_code_local = stn(h_code_local, transf_matrices_inv[:, idx], (h_code_local.shape[0], h_code_local.shape[1], 16, 16))
            h_code_locals += h_code_local

        h_code = self.conv1(image)
        h_code = self.act(h_code)
        h_code = self.conv2(h_code)
        h_code = self.bn2(h_code)
        h_code = self.act(h_code)

        # combine local and global
        h_code = torch.cat((h_code, h_code_locals), 1)

        h_code = self.conv3(h_code)
        h_code = self.bn3(h_code)
        h_code = self.act(h_code)

        h_code = self.conv4(h_code)
        h_code = self.bn4(h_code)
        h_code = self.act(h_code)
        return h_code

    def forward(self, image, label, transf_matrices, transf_matrices_inv):
        img_embedding = self._encode_img(image, label, transf_matrices, transf_matrices_inv)

        return img_embedding
