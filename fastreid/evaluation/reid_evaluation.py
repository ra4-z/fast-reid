# encoding: utf-8
# modified by Shengyuan
'''
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
'''
import copy
from lib2to3.pytree import convert
import logging
import time
import itertools
from collections import OrderedDict

import numpy as np
from regex import D
from sympy import true
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

    def evaluate(self, vis=False): 
        logger = logging.getLogger(__name__)
        logger.info('Evaluating results')
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
            logger.info('Test with AQE setting')
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)

        if self.cfg.TEST.RERANK.ENABLED:
            logger.info('Test with rerank setting')
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == 'cosine':
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric='jaccard', k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        # calculate cmc, ap
        from .rank import evaluate_rank
        cmc, all_AP, all_INP, rank_id_mat = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)

        ### visualize rank_id_mat
        if vis:
            vis_res_imgs(rank_id_mat, dist, query_img_paths, gallery_img_paths)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results['metric'] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results['TPR@FPR={:.0e}'.format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info('> compiling reid evaluation cython tool')

                compile_helper()

                logger.info(
                    '>>> done with reid evaluation cython tool. Compilation time: {:.3f} '
                    'seconds'.format(time.time() - start_time))
        comm.synchronize()




def vis_res_imgs(rank_id_mat, dist, q_pics, g_pics, res_dir='/data/codes/fast-reid/res/',
                 root_dir='/data/codes/fast-reid/',only_neg=True):
    logger = logging.getLogger(__name__)
    logger.info('visualizing results')
    # copy src and res pictures
    import cv2
    text_side_len = 10
    text_pos = (0, text_side_len) #(x,y) of left bottom corner of text
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = text_side_len//10
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
                img = cv2.rectangle(img,(0,0),(w,h),(255,0,0),2) # blue for itself
                img = cv2.putText(img, q_id, text_pos, font, font_size, font_color, font_thickness)
                concat = img
            else:
                # get gallery pic id
                g_id = g_pics[ranks[j-1]].split('/')[-1].split('_')[0]
                if g_id==q_id: # the right pic
                    img = cv2.rectangle(img,(0,0),(w,h),(0,255,0),2) # green for the right
                else: # the wrong pic
                    img = cv2.rectangle(img,(0,0),(w,h),(0,0,255),2) # red for the wrong
                img = cv2.putText(img, g_id, text_pos, font, font_size, font_color, font_thickness) # id
                img = cv2.putText(img, f'{dist[i][ranks[j-1]]:0.4f}', (0,h), font, font_size, font_color, font_thickness) # dist
                concat = cv2.hconcat([concat,img])

        cv2.imwrite(res_dir+'res_'+q_pics[i].split('/')[-1],concat)

def gen_gallery(gallery_features: torch.Tensor,
                gallery_pids,
                gallery_camids,
                gallery_direcs,
                gallery_frameids,
                gallery_coverages=None,
                gallery_confs=None,):
    '''
        function: 
            1) average the gallery features of the same object in the same camera
            2) get the direction of the last frame of each id in each camera
        return: new_gallery_features,new_gallery_pids,new_gallery_camids,new_gallery_direcs
    '''
    
    new_gallery_features,new_gallery_pids,new_gallery_camids,new_gallery_direcs = [],[],[],[]
    
    keep_num_ref = 3
    
    # save gallery feature in camid-pid-feature format
    gallery_feature_dict = dict()
    gallery_feature_dict_frameid = dict()
    for idx, feature in enumerate(gallery_features):
        pid = gallery_pids[idx]
        camid = gallery_camids[idx]
        frameid = gallery_frameids[idx]
        if camid not in gallery_feature_dict:
            gallery_feature_dict[camid] = dict()
            gallery_feature_dict_frameid[camid] = dict()
        if pid not in gallery_feature_dict[camid]:
            gallery_feature_dict[camid][pid] = []
            gallery_feature_dict_frameid[camid][pid] = []
        gallery_feature_dict[camid][pid].append(feature)
        gallery_feature_dict_frameid[camid][pid].append(frameid)
    
    # save coverages and confs in camid-pid-data format
    if gallery_coverages is not None:
        gallery_coverage_dict = dict()
        gallery_conf_dict = dict()
        for idx, coverage in enumerate(gallery_coverages):
            pid = gallery_pids[idx]
            camid = gallery_camids[idx]
            if camid not in gallery_coverage_dict:
                gallery_coverage_dict[camid] = dict()
                gallery_conf_dict[camid] = dict()
            if pid not in gallery_coverage_dict[camid]:
                gallery_coverage_dict[camid][pid] = []
                gallery_conf_dict[camid][pid] = []
            gallery_coverage_dict[camid][pid].append(coverage)
            gallery_conf_dict[camid][pid].append(gallery_confs[idx])
    
    # save the last direction of each id in each camera, in camid-pid-direction format
    gallery_direc_dict = dict()
    for idx,direc in enumerate(gallery_direcs):
        pid = gallery_pids[idx]
        camid = gallery_camids[idx]
        frameid = gallery_frameids[idx]
        if camid not in gallery_direc_dict:
            gallery_direc_dict[camid] = dict()
        if pid not in gallery_direc_dict[camid]:
            gallery_direc_dict[camid][pid] = []
        if gallery_direc_dict[camid][pid] == []:
            gallery_direc_dict[camid][pid]=(frameid,direc)
        elif gallery_direc_dict[camid][pid][0] < frameid: # record the last direction
            gallery_direc_dict[camid][pid]=(frameid,direc)
            
    
    for camid in gallery_feature_dict:
        for pid in gallery_feature_dict[camid]:
            # calculate the gallery features of the same object in the same camera
            features = torch.stack(gallery_feature_dict[camid][pid],dim=0) # check var?
            if gallery_coverages is not None:
                cover_revert = 1-torch.vstack(gallery_coverage_dict[camid][pid]).to(dtype=torch.float32)
                conf = torch.vstack(gallery_conf_dict[camid][pid]).to(dtype=torch.float32)
                features = features*cover_revert*conf
            # ------ for outliers of tracker -------
            if features.shape[0] > keep_num_ref: 
                keep = np.argpartition(gallery_feature_dict_frameid[camid][pid], -keep_num_ref)[:-keep_num_ref] # get the last 3 frameids
                features = features[keep]
            # ----------------------------------------
            new_feature = torch.mean(features,dim=0)
            new_gallery_features.append(new_feature)
            # save the direction of the last frame of each id in each camera
            direc = gallery_direc_dict[camid][pid][1]
            new_gallery_direcs.append(direc)
            
            new_gallery_camids.append(camid)
            new_gallery_pids.append(pid)
    
    new_gallery_features = torch.stack(new_gallery_features, dim=0) # torch.float32
    new_gallery_direcs = torch.stack(new_gallery_direcs, dim=0)
    
    return new_gallery_features,new_gallery_pids,new_gallery_camids,new_gallery_direcs,
    
    
def new_match(dist, q_pids, q_camids, q_dirs, g_pids, g_camids, g_dirs, limit_dir=False,device='cuda'):
    '''
        function: find the best match of each query, and return the accuracy
    '''
    # remove the same camera
    
    
    start_time = time.perf_counter()
    inf = float('inf')
    q_num, g_num = len(q_pids), len(g_pids)
    q_camids = torch.tensor(q_camids, dtype=torch.int8).to(device)
    q_camids = q_camids.tile((g_num,1)).t()
    g_camids = torch.tensor(g_camids, dtype=torch.int8).to(device)
    g_camids = g_camids.t().tile((q_num,1))
    remove = (q_camids == g_camids)
    
    # remove = np.array([[inf if g_camid == q_camid else 1. for g_camid in g_camids] for q_camid in q_camids])
    now_time = time.perf_counter()
    logger.info(f'remove the same camera time cost: {now_time-start_time:.4f}s')
    
    # different directions mean no chance to match
    if limit_dir:
        start_time = time.perf_counter()
        q_dirs = torch.tensor(q_dirs, dtype=torch.float16).to(device)
        g_dirs = torch.tensor(g_dirs, dtype=torch.float16).to(device)
        q_g_dirs = torch.mm(q_dirs,g_dirs.t())
        remove2 = (q_g_dirs < 0)
        remove = remove | remove2    
        # remove2 = np.array([[inf if g_dir.dot(q_dir)<0 else 1. for g_dir in g_dirs] for q_dir in q_dirs])
        # remove = remove*remove2
        now_time = time.perf_counter()
        logger.info(f'remove the wrong direction time cost: {now_time-start_time:.4f}s')
    
    
    start_time = time.perf_counter()
    # dist = dist + remove
    # dist = dist * remove
    remove = remove.to(dtype=torch.float16)*1000
    dist = torch.tensor(dist, dtype=torch.float16).to(device)
    dist = dist + remove
    now_time = time.perf_counter()
    logger.info(f'revise distance matrix time cost: {now_time-start_time:.4f}s')
    
    start_time = time.perf_counter()
    rank_id = torch.argmin(dist,axis=1).cpu().numpy()
    now_time = time.perf_counter()
    logger.info(f'ranking distance matrix time cost: {now_time-start_time:.4f}s')
    
    # calculate acc
    start_time = time.perf_counter()
    pred = np.array(g_pids)[rank_id]
    acc = np.equal(q_pids, pred).sum() / len(q_pids)
    now_time = time.perf_counter()
    logger.info(f'calculate accuaracy time cost: {now_time-start_time:.4f}s')
    
    return acc, q_pids, pred
    
        
def new_vis(q_pids, g_pids, res_dir='/data/codes/fast-reid/res/',
            dataset_dir='/data/codes/fast-reid/datasets/overpass',
            only_neg=False):
    
    def find_pic(path, pid, camid=None):
        for file in os.listdir(path):
            if file.startswith(f"{pid:04d}"):
                if camid != None:
                    this_camid = file.split('_')[1][1:]
                    if this_camid==camid:
                        continue
                return os.path.join(path, file)
        return None
    
    logger = logging.getLogger(__name__)
    logger.info('visualizing results')
    # copy src and res pictures
    import cv2
    text_side_len = 10
    text_pos = (0, text_side_len) #(x,y) of left bottom corner of text
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = text_side_len//10
    font_color = (0,0,0)
    font_thickness = 2
    import os
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    else:
        os.system(f'rm -rf {res_dir}')
        os.makedirs(res_dir)
    
    for idx in range(len(q_pids)):
        if only_neg and q_pids[idx]==g_pids[idx]:
            continue
        
        q_pic_name = find_pic(os.path.join(dataset_dir,'image_query'), q_pids[idx])
        q_camid = q_pic_name.split('_')[1][1:]
        g_pic_name = find_pic(os.path.join(dataset_dir,'image_test'), g_pids[idx],q_camid)
        if g_pic_name==None:
            continue
        q_pic = cv2.imread(q_pic_name)
        g_pic = cv2.imread(g_pic_name)
        
        ## -------- temporily -----------
        if not os.path.exists('/data/codes/fast-reid/res/true/'):
            os.makedirs('/data/codes/fast-reid/res/true/')
        if not os.path.exists('/data/codes/fast-reid/res/wrong/'):
            os.makedirs('/data/codes/fast-reid/res/wrong/')
        if q_pids[idx]==g_pids[idx]:
            tmp_name = '/data/codes/fast-reid/res/true/'+f'res_{q_pids[idx]:04d}.jpg'
            cv2.imwrite(tmp_name,q_pic)
        else:
            tmp_name = '/data/codes/fast-reid/res/wrong/'+f'res_{q_pids[idx]:04d}.jpg'
            cv2.imwrite(tmp_name,q_pic)
        ## ------------------------------    
            
        
        hq,wq,cq = q_pic.shape
        hg,wg,cg = g_pic.shape
        h,w = max(hq,hg), max(wq,wg)
        q_pic = cv2.copyMakeBorder(q_pic, 0, h-hq, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        q_pic = cv2.rectangle(q_pic,(0,0),(w,h),(255,0,0),2) # blue for itself
        q_pic = cv2.putText(q_pic, str(q_pids[idx]), text_pos, font, font_size, font_color, font_thickness)
        g_pic = cv2.copyMakeBorder(g_pic, 0, h-hg, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        if q_pids[idx]==g_pids[idx]:
            g_pic = cv2.rectangle(g_pic,(0,0),(w,h),(0,255,0),2) # green for the right
        else:
            g_pic = cv2.rectangle(g_pic,(0,0),(w,h),(0,0,255),2) # red for the wrong
        g_pic = cv2.putText(g_pic, str(g_pids[idx]), text_pos, font, font_size, font_color, font_thickness)
        concat = cv2.hconcat([q_pic,g_pic])
        cv2.imwrite(res_dir+f'res_{q_pids[idx]:04d}.jpg',concat)
        
        
class MyReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self._cpu_device = torch.device('cpu')
        
        self._cuda_device = torch.device('cuda') if torch.cuda.is_available() else self._cpu_device
        
        self._predictions = []
        self._compile_dependencies()

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        prediction = {
            'feats': outputs.to(self._cpu_device, torch.float32), # TODO： torch.float16
            'pids': inputs['targets'].to(self._cpu_device),
            'camids': inputs['camids'].to(self._cpu_device),
            'frameids': inputs['frameids'].to(self._cpu_device),
            'img_paths': inputs['img_paths'], #list
            'direcs': inputs['direcs'].to(self._cpu_device), #list TODO:torch.tensor
            'coverages':inputs['coverages'].to(self._cpu_device), #list TODO:torch.tensor
            'confs':inputs['confs'].to(self._cpu_device), #list TODO:torch.tensor
        }
        self._predictions.append(prediction)

    @torch.no_grad()
    def evaluate(self, vis=True): 
        logger = logging.getLogger(__name__)
        logger.info('Evaluating results')
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
        frameids = []
        img_paths = []
        direcs = []
        coverages = []
        confs = []
        # combine all
        for prediction in predictions:
            features.append(prediction['feats'])
            pids.append(prediction['pids'])
            camids.append(prediction['camids'])
            frameids.append(prediction['frameids'])
            img_paths.extend(prediction['img_paths'])
            direcs.extend(prediction['direcs'])
            coverages.extend(prediction['coverages'])
            confs.extend(prediction['confs'])

        features = torch.cat(features, dim=0).to(dtype=torch.float16, device=self._cuda_device) # shape: (num_query+num_gallery, 2048)
        # TODO: directly to numpy
        pids = torch.cat(pids, dim=0).numpy()  
        camids = torch.cat(camids, dim=0).numpy()
        frameids = torch.cat(frameids, dim=0).numpy()
        # -------------
        img_paths = img_paths
        direcs = torch.vstack(direcs) # shape: (num_query+num_gallery, 3)
        coverages = torch.vstack(coverages) # shape: (num_query+num_gallery, 1)
        confs = torch.vstack(confs) # shape: (num_query+num_gallery, 1)

        # separating query and gallery
        # query feature, person ids and camera ids
        query_features = features[:self._num_query] 
        query_pids = pids[:self._num_query]
        query_camids = camids[:self._num_query]
        query_img_paths = img_paths[:self._num_query]
        query_frameids = frameids[:self._num_query]
        query_direcs = direcs[:self._num_query]
        
        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = pids[self._num_query:]
        gallery_camids = camids[self._num_query:]
        gallery_img_paths = img_paths[self._num_query:]
        gallery_frameids = frameids[self._num_query:]
        gallery_direcs = direcs[self._num_query:]
        gallery_coverages = coverages[self._num_query:]
        gallery_confs = confs[self._num_query:]

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED: # rerank
            logger.info('Test with AQE setting')
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)


        start_time = time.perf_counter()
        
        # new_gallery_features,new_gallery_pids,new_gallery_camids,new_gallery_direcs = \
        #     gen_gallery(gallery_features,gallery_pids,gallery_camids, 
        #                 gallery_direcs,gallery_frameids,
        #                 gallery_coverages,gallery_confs)
        new_gallery_features,new_gallery_pids,new_gallery_camids,new_gallery_direcs = \
            gen_gallery(gallery_features,gallery_pids,gallery_camids, 
                        gallery_direcs,gallery_frameids)
        now_time = time.perf_counter()
        logger.info(f'Unifying features from {len(gallery_pids)} to {len(new_gallery_pids)}, \
                    time cost: {now_time-start_time:.4f}s')
        
        start_time = time.perf_counter()
        dist = build_dist(query_features, new_gallery_features, self.cfg.TEST.METRIC)
        now_time = time.perf_counter()
        logger.info(f'Calculating distance matrix {len(query_pids)}x{len(new_gallery_pids)}, time cost: {now_time-start_time:.4f}s')
        
        start_time = time.perf_counter()
        rank1, q_pids, pred = new_match(dist, query_pids, query_camids, query_direcs, 
                          new_gallery_pids, new_gallery_camids, new_gallery_direcs,
                          limit_dir=True)
        now_time = time.perf_counter()
        logger.info(f'Matching time cost: {now_time-start_time:.4f}s')
        logger.info(f'rank1: {rank1:0.4f}')
        
        ##------temporily-----
        # import json
        # save_path = "/data/codes/fast-reid/gt_pred.txt"
        # res = dict(zip(query_pids,pred))
        # with open(save_path,'w') as wj:
        #     json.dump(res,wj)
        
        ##---------------------
        
        ##------temporily-----
        '''
        import json
        save_res = dict() # camid-gt-pred
        for idx in range(len(q_pids)):
            if str(query_camids[idx]) not in save_res:
                save_res[str(query_camids[idx])] = dict()
            if str(q_pids[idx]) not in save_res[str(query_camids[idx])]:
                save_res[str(query_camids[idx])][str(q_pids[idx])] = None
            save_res[str(query_camids[idx])][str(q_pids[idx])] = str(pred[idx])
        save_path = "/data/codes/fast-reid/gt_pred.json"
        with open(save_path,'w') as wj:
            json.dump(save_res,wj)
        '''
        
        ##--------------------
        
        start_time = time.perf_counter()
        new_vis(q_pids=q_pids, g_pids=pred, only_neg=True)
        now_time = time.perf_counter()
        logger.info(f'visualization time cost: {now_time-start_time:.4f}s')
        
        # old version
        '''
        dist = build_dist(query_features, gallery_features, self.cfg.TEST.METRIC)
        logger.info('Distance calculation time: {}'.format(time.perf_counter() - start_time))
        
        if self.cfg.TEST.RERANK.ENABLED:
            logger.info('Test with rerank setting')
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA

            if self.cfg.TEST.METRIC == 'cosine':
                query_features = F.normalize(query_features, dim=1)
                gallery_features = F.normalize(gallery_features, dim=1)

            rerank_dist = build_dist(query_features, gallery_features, metric='jaccard', k1=k1, k2=k2)
            dist = rerank_dist * (1 - lambda_value) + dist * lambda_value

        
        
        # calculate cmc, ap
        from .rank import my_evaluate_rank
        cmc, all_AP, all_INP, rank_id_mat = \
            my_evaluate_rank(dist, query_pids, gallery_pids, 
                             query_camids, gallery_camids,
                             query_frameids, gallery_frameids,)

        ### visualize rank_id_mat
        if vis:
            vis_res_imgs(rank_id_mat, dist, query_img_paths, gallery_img_paths)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1] * 100
        self._results['mAP'] = mAP * 100
        self._results['mINP'] = mINP * 100
        self._results['metric'] = (mAP + cmc[0]) / 2 * 100

        if self.cfg.TEST.ROC.ENABLED:
            from .roc import evaluate_roc
            scores, labels = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results['TPR@FPR={:.0e}'.format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)
        '''
        

    def _compile_dependencies(self):
        # Since we only evaluate results in rank(0), so we just need to compile
        # cython evaluation tool on rank(0)
        if comm.is_main_process():
            try:
                from .rank_cylib.rank_cy import evaluate_cy
            except ImportError:
                start_time = time.time()
                logger.info('> compiling reid evaluation cython tool')

                compile_helper()

                logger.info(
                    '>>> done with reid evaluation cython tool. Compilation time: {:.3f} '
                    'seconds'.format(time.time() - start_time))
        comm.synchronize()

