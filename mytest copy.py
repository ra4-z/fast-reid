# -*- coding: utf-8 -*-

from __future__ import print_function, division
import datetime
# ------
from MyDataloader import MyDataset
from torch.utils.data import Dataset, DataLoader
from MyEval import extract_info, calculate_result
from rank import evaluate_rank
from distance import compute_distance_matrix
# ------
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from utils import load_network
from tqdm import tqdm
from model import ft_net, ft_net_angle, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_arc, ft_net_IR, ft_net_SE, ft_net_DSE, ft_net_dense, ft_net_NAS, PCB, PCB_test, CPB
# from evaluate_gpu import calculate_result
from evaluate_rerank import calculate_result_rerank


# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--ms', default='1', type=str,
                    help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--which_epoch', default='last',
                    type=str, help='0,1,2,3...or last')
parser.add_argument(
    '--test_dir', default='/home/AICIty-reID-2020/pytorch/data/', type=str, help='./test_data')
parser.add_argument('--name', default='SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug',
                    type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='save model path')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')
parser.add_argument('--inputsize', default=320, type=int, help='batchsize')
parser.add_argument('--h', default=320, type=int, help='batchsize')
parser.add_argument('--w', default=320, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use densenet121')
parser.add_argument('--use_SE', action='store_true', help='use densenet121')
parser.add_argument('--use_EF4', action='store_true', help='use densenet121')
parser.add_argument('--use_EF5', action='store_true', help='use densenet121')
parser.add_argument('--use_EF6', action='store_true', help='use densenet121')
parser.add_argument('--use_DSE', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join(
    '/home/AICIty-reID-2020/pytorch', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.CPB = config['CPB']
opt.inputsize = config['inputsize']
opt.stride = config['stride']
opt.angle = config['angle']
opt.use_EF4 = config['use_EF4']
opt.use_EF5 = config['use_EF5']
opt.use_EF6 = config['use_EF6']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'pool' in config:
    opt.pool = config['pool']

opt.use_dense = config['use_dense']
if 'use_NAS' in config:  # compatible with early config
    opt.use_NAS = config['use_NAS']
else:
    opt.use_NAS = False

if 'use_SE' in config:  # compatible with early config
    opt.use_SE = config['use_SE']
else:
    opt.use_SE = False

if 'use_DSE' in config:  # compatible with early config
    opt.use_DSE = config['use_DSE']
else:
    opt.use_DSE = False

if 'use_IR' in config:  # compatible with early config
    opt.use_IR = config['use_IR']
else:
    opt.use_IR = False


if 'arc' in config:
    opt.arc = config['arc']
else:
    opt.arc = False

opt.nclasses = config['nclasses']

#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# ---------load data---------
data_transforms = transforms.Compose([
    transforms.Resize((round(opt.h*1.1), round(opt.w*1.1)), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir

# # image_datasets['gallery'].imgs[0] -> ('/home/AICIty-reID-20...8_0009.jpg', 0)
# image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms) for x in [
#     'gallery', 'query']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                               shuffle=False, num_workers=20) for x in ['gallery', 'query']}


mydataset = MyDataset(transform=data_transforms)
dataloader = DataLoader(mydataset, batch_size=16)


use_gpu = torch.cuda.is_available()

# -----------Load model-----------
model, _, epoch = load_network(opt.name, opt)
model = model.eval()
if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

# ---------Extract feature---------
features, camids, labels, pic_names = extract_info(model, dataloader)
query_feature = gallery_feature = features
query_feature = torch.FloatTensor(query_feature).cuda()
gallery_feature = torch.FloatTensor(gallery_feature).cuda()
dist_mat = compute_distance_matrix(query_feature, gallery_feature)
dist_mat = dist_mat.cpu().numpy()

all_cmc, all_AP, all_INP = evaluate_rank(
    dist_mat, labels, labels, camids, camids, 10)
mAP = np.mean(all_AP)
mINP = np.mean(all_INP)
print(f"mAP:{mAP}\t R1:{all_cmc[0]}\t R5:{all_cmc[4]}\t R10:{all_cmc[9]}")

# cmc, mAP = calculate_result(features, labels, features, labels)
