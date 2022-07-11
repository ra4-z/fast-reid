# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import sys
sys.path.append("./aicity2020/")
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
import scipy.io
import yaml
from utils import load_network
from tqdm import tqdm
from model import ft_net, ft_net_angle, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_arc, ft_net_IR, ft_net_SE, ft_net_DSE, ft_net_dense, ft_net_NAS, PCB, PCB_test, CPB


def aicity_2020_model():

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--ms', default='1', type=str,
                        help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
    parser.add_argument('--which_epoch', default='last',
                        type=str, help='0,1,2,3...or last')
    parser.add_argument(
        '--test_dir', default='/home/AICIty-reID-2020/pytorch/data/', type=str, help='./test_data')
    parser.add_argument('--name', default= 'ft_Res50_imbalance_s1_256_p0.5_lr1_mt_d0.2_b48_w5',
                            # 'SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug', 
                            # 'ft_Res50_imbalance_s1_256_p0.5_lr1_mt_d0.2_b48_w5',
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


    use_gpu = torch.cuda.is_available()

    # -----------Load model-----------
    model, _, epoch = load_network(opt.name, opt)
    model = model.eval()
    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    
    return model

if __name__=='__main__':
    model = aicity_2020_model()

