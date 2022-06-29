# -*- coding: utf-8 -*-

from __future__ import print_function, division
import datetime
from distance import *

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
from evaluate_gpu import calculate_result
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

# image_datasets['gallery'].imgs[0] -> ('/home/AICIty-reID-20...8_0009.jpg', 0)
image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms) for x in [
    'gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=False, num_workers=20) for x in ['gallery', 'query']}
use_gpu = torch.cuda.is_available()

# -----------Load model-----------
model, _, epoch = load_network(opt.name, opt)
model = model.eval()
if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

# ---------Extract feature---------


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 2048).zero_().cuda()

        for i in range(1):
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(
                        input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                outputs = model(input_img)
                if opt.CPB:
                    outputs = outputs[1]

                # print(outputs.size())
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        elif opt.CPB:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(4)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        # print(ff.shape)
        features = torch.cat((features, ff.data.cpu().float()), 0)  # catenate
    return features


with torch.no_grad():
    model = model.eval()

    ##
    access_start = datetime.datetime.now()
    access_start_str = access_start.strftime('%Y-%m-%d %H:%M:%S')
    ##
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    ##
    access_end = datetime.datetime.now()
    access_end_str = access_end.strftime('%Y-%m-%d %H:%M:%S')
    access_delta = (access_end-access_start).seconds*1000
    print(f"time for inference {access_delta}")
    ##
    # query_feature = extract_feature(model,dataloaders['query'])
    query_feature = gallery_feature


# calculate distance
##
access_start = datetime.datetime.now()
access_start_str = access_start.strftime('%Y-%m-%d %H:%M:%S')
##
distance_mat = compute_distance_matrix(gallery_feature, query_feature)
distance_mat = distance_mat.numpy()

##
access_end = datetime.datetime.now()
access_end_str = access_end.strftime('%Y-%m-%d %H:%M:%S')
access_delta = (access_end-access_start).seconds*1000
print(f"time for distance {access_delta}")
##


k = 10
topk = np.argsort(distance_mat, axis=1)[:, :k]
res = []
res_root = '/home/AICIty-reID-2020/pytorch/res/'
src_root = '/home/AICIty-reID-2020/pytorch/data/cars/'
for i, line in enumerate(topk):
    src_pic = image_datasets['gallery'].imgs[i][0]
    pic_name = src_pic.split('/')[-1][:-4]
    path = res_root+pic_name+'/'

    if not os.path.exists(path):
        os.makedirs(path)

    os.system(f'cp {src_pic} {path}src.jpg')

    for j, idx in enumerate(line):
        res_pic = image_datasets['gallery'].imgs[idx][0]
        res_pic_name = str(j)+res_pic.split('/')[-1]
        os.system(f'cp {res_pic} {path}{res_pic_name}')


# image_datasets['gallery'].imgs[0] -> ('/home/AICIty-reID-20...8_0009.jpg', 0)
print("------------done----------")
