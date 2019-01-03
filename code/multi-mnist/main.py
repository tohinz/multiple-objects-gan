from __future__ import print_function
import torch.backends.cudnn as cudnn
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil
import dateutil.tz
from shutil import copyfile


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from miscc.datasets import TextDataset
from miscc.config import cfg, cfg_from_file
from miscc.utils import mkdir_p
from trainer import GANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='birds_stage1.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../../output/%s_%s' % \
                 (cfg.DATASET_NAME, timestamp)

    cudnn.benchmark = True

    num_gpu = len(cfg.GPU_ID.split(','))
    if cfg.TRAIN.FLAG:
        try:
            os.makedirs(output_dir)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise

        copyfile(sys.argv[0], output_dir + "/" + sys.argv[0])
        copyfile("trainer.py", output_dir + "/" + "trainer.py")
        copyfile("model.py", output_dir + "/" + "model.py")
        copyfile("miscc/utils.py", output_dir + "/" + "utils.py")
        copyfile("miscc/datasets.py", output_dir + "/" + "datasets.py")
        copyfile(args.cfg_file, output_dir + "/" + "cfg_file.yml")

        imsize=64

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = TextDataset(cfg.DATA_DIR, split="train", imsize=imsize, transform=img_transform,
                              crop=True)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

        algo = GANTrainer(output_dir)
        algo.train(dataloader)
    else:
        datapath = os.path.join(cfg.DATA_DIR, "test")
        algo = GANTrainer(output_dir)
        algo.sample(datapath, num_samples=25, draw_bbox=True)
