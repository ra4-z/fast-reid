# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
import json

@DATASET_REGISTRY.register()
class Overpass(ImageDataset):
    """VeRi modified.

    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.

    URL: `<https://vehiclereid.github.io/VeRi/>`_

    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "overpass"
    dataset_name = "overpass"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.json_path = osp.join(self.dataset_dir, 'all_data.json')
        
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.json_path,
        ]
        self.check_before_run(required_files)

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        # for saving memory
        del self.data
        
        super(Overpass, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d)_([\d]+)') # 需匹配所有信息

        data = []
        for img_path in img_paths:# 需要按照文件名顺序。不用。
            pid, camid, frameid = map(int, pattern.search(img_path).groups()) # map(function, iterable, ...)
            if pid == -1: continue  # junk images are just ignored
            
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            
            # wrong when 'datasets/overpass/image_query/0054_c00_0246_0072.jpg'

            loc = self.data[str(camid)][str(frameid)][str(pid)]["loc"]
            conf = self.data[str(camid)][str(frameid)][str(pid)]["conf"]
            cover = self.data[str(camid)][str(frameid)][str(pid)]["cover"]
            size = self.data[str(camid)][str(frameid)][str(pid)]["size"]
            direc = self.data[str(camid)][str(frameid)][str(pid)]["direc"]
            data.append((img_path, pid, camid, frameid, loc, conf, cover, size, direc))

        return data
