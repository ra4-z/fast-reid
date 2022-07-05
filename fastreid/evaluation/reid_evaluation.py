# encoding: utf-8
# modified by Shengyuan
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from fastreid.utils import comm
from fastreid.utils.compute_dist import build_dist
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank_cylib import compile_helper

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')

        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32),
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'img_paths': inputs['img_paths'] #list
        }
        self._predictions.append(prediction)

    def evaluate(self, vis=True): #TODO: this is eval key part! find all used tricks!
        logger = logging.getLogger(__name__)
        logger.info("Evaluating results")
        if comm.get_world_size() > 1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}

        else:
            predictions = self._predictions

        features = []
        pids = []
        camids = []
        img_paths = []
        # combine all
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            img_paths.extend(prediction['img_paths'])

        features = torch.cat(features, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        img_paths = img_paths

        # separating query and gallery
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]
        query_img_paths = img_paths[:self._num_query]

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]
        gallery_img_paths = img_paths[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED: # rerank
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        ## ----for testing----
        # import pandas as pd
        # intermidiate_res_path = '/home/fast-reid/intermidiate_res/'
        # pd.DataFrame(dist[::3,::21]).to_excel(intermidiate_res_path+"dist_mat_VeRi_sbs_R50-ibn_VeRi.xlsx",index=False)
        # print("dist saved")
        # pd.DataFrame(gallery_pids).to_excel(intermidiate_res_path+"gallery_pids_VeRi_sbs_R50-ibn_VeRi.xlsx")
        # pd.DataFrame(query_pids).to_excel(intermidiate_res_path+"query_pids_VeRi_sbs_R50-ibn_VeRi.xlsx")
        # pd.DataFrame(gallery_camids).to_excel(intermidiate_res_path+"gallery_camids_VeRi_sbs_R50-ibn_VeRi.xlsx")
        # pd.DataFrame(query_camids).to_excel(intermidiate_res_path+"query_camids_VeRi_sbs_R50-ibn_VeRi.xlsx")
        ## ------------------

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == "cosine":
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric="jaccard", k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        # calculate cmc, ap
        from .rank import evaluate_rank
        cmc, all_AP, all_INP, rank_id_mat = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        ### visualize rank_id_mat
        if vis:
            vis_res_imgs(rank_id_mat, query_img_paths, gallery_img_paths)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results["metric"] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info("> compiling reid evaluation cython tool")

                compile_helper()

                logger.info(
                    ">>> done with reid evaluation cython tool. Compilation time: {:.3f} "
                    "seconds".format(time.time() - start_time))
        comm.synchronize()




def vis_res_imgs(rank_id_mat, q_pics, g_pics, res_dir='/home/fast-reid/res/',root_dir='/home/fast-reid/'):
    logger = logging.getLogger(__name__)
    logger.info("visualizing results")
    # copy src and res pictures
    import cv2
    text_pos = (15, 15)
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1
    font_color = (0,0,0)
    font_thickness = 2
    import os
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for i,ranks in enumerate(rank_id_mat):
        # invalid rank
        if ranks[0]==-1:
            continue
        # get query and gallery picutres
        concat = None
        pics_to_concat = []
        pics_to_concat.append(q_pics[i])
        height = cv2.imread(root_dir+q_pics[i]).shape[0]
        for rank in ranks:
            pics_to_concat.append(g_pics[rank])
            h = cv2.imread(root_dir+g_pics[rank]).shape[0]
            height = max(h,height)
        
        # get query pic id
        q_id = q_pics[i].split('/')[-1].split('_')[0]
        # concat pictures
        for j,pic in enumerate(pics_to_concat):
            img = cv2.imread(root_dir+pic)
            h,w,c = img.shape
            img = cv2.copyMakeBorder(img,0,height-h,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            
            if j == 0: # src pic
                img = cv2.rectangle(img,(0,0),(w,h),(255,0,0),2) # itself
                img = cv2.putText(img, q_id, text_pos, font, font_size, font_color, font_thickness)
                concat = img
            else:
                # get gallery pic id
                g_id = g_pics[ranks[j-1]].split('/')[-1].split('_')[0]
                if g_id==q_id: # the right pic
                    img = cv2.rectangle(img,(0,0),(w,h),(0,255,0),2) 
                else: # the wrong pic
                    img = cv2.rectangle(img,(0,0),(w,h),(0,0,255),2) 
                img = cv2.putText(img, g_id, text_pos, font, font_size, font_color, font_thickness)
                concat = cv2.hconcat([concat,img])

        cv2.imwrite(res_dir+'res_'+q_pics[i].split('/')[-1],concat)
