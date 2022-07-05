from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.config import get_cfg
from MyEval import extract_info, calculate_result
from torch.utils.data import Dataset, DataLoader
from MyDataloader import MyDataset
from torchvision import datasets, models, transforms
import torch
import sys
from rank import evaluate_rank
from distance import compute_distance_matrix
import numpy as np
sys.path.append('.')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

intermidiate_res_path = './intermidiate_res/'
mydata = "/home/data/frames20220628/"

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        model.eval()
        # res = DefaultTrainer.test(cfg, model)
        # return res

        data_transforms = transforms.Compose([
            transforms.Resize((256, 256), interpolation=3), # 一模一样
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # fast-reid没做这个 
        ])
        mydataset = MyDataset(mydata, transform=data_transforms)
        dataloader = DataLoader(mydataset, batch_size=16)

        features, camids, labels, pic_names = extract_info(model, dataloader)

        query_feature = gallery_feature = features
        query_feature = torch.FloatTensor(query_feature).cuda()
        gallery_feature = torch.FloatTensor(gallery_feature).cuda()
        dist_mat = compute_distance_matrix(query_feature, gallery_feature)
        dist_mat = dist_mat.cpu().numpy()
        

        import pandas as pd
        np.savetxt(intermidiate_res_path+'dist_mat.txt', dist_mat)
        pd.DataFrame(dist_mat).to_excel(intermidiate_res_path+"dist_mat.xlsx",index=False)
        pd.DataFrame(camids).to_excel(intermidiate_res_path+"camid.xlsx")
        pd.DataFrame(labels).to_excel(intermidiate_res_path+"label.xlsx")
        pd.DataFrame(pic_names).to_excel(intermidiate_res_path+"pic_name.xlsx")
        
        # import os
        # k = 10
        # topk = np.argsort(dist_mat, axis=1)[:, :k]
        # res = []
        # res_root = '/home/fast-reid/res20220628/'
        # src_root = '/home/data/frames20220628/'
        # if not os.path.exists(res_root):
        #     os.makedirs(res_root)
        # for i, line in enumerate(topk):
        #     src_pic = src_root+pic_names[line[0]]
        #     res_pic_path = res_root+pic_names[line[0]][:-5]
        #     if not os.path.exists(res_pic_path):
        #         os.makedirs(res_pic_path)
        #     os.system(f'cp {src_pic} {res_pic_path}/src.jpg')

        #     for j, idx in enumerate(line):
        #         res_pic = src_root + pic_names[idx]
        #         rest_res_pic_path = res_pic_path+'/'+str(j)+"_"+pic_names[idx]
        #         os.system(f'cp {res_pic} {rest_res_pic_path}')

        


        
        all_cmc, all_AP, all_INP, rank_id_mat = evaluate_rank(
            dist_mat, labels, labels, camids, camids, 10)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        print(
            f"mAP:{mAP:.4f}\t R1:{all_cmc[0]:.4f}\t R5:{all_cmc[4]:.4f}\t R10:{all_cmc[9]:.4f}")
        
        # # copy src and res pictures
        # import cv2
        # text_pos = (10, 10)
        # font = cv2.FONT_HERSHEY_PLAIN
        # font_size = 0.5
        # font_color = (0,0,0)
        # font_thickness = 1
        # import os
        # res_dir = '/home/fast-reid/res20220630/'
        # if not os.path.exists(res_dir):
        #     os.makedirs(res_dir)
        # for i,ranks in enumerate(rank_id_mat):
        #     concat = None
        #     pics_to_concat = []
        #     pics_to_concat.append(pic_names[i])
        #     height = cv2.imread(mydata+pic_names[i]).shape[0]
        #     for rank in ranks:
        #         pics_to_concat.append(pic_names[rank])
        #         h = cv2.imread(mydata+pic_names[rank]).shape[0]
        #         height = max(h,height)
            
        #     for j,pic in enumerate(pics_to_concat):
        #         img = cv2.imread(mydata+pic)
        #         h,w,c = img.shape
        #         img = cv2.copyMakeBorder(img,0,height-h,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        #         if j == 0:
        #             img = cv2.rectangle(img,(0,0),(w,h),(255,0,0),2) # itself
        #             img = cv2.putText(img, pic_names[i][:3], text_pos, font, font_size, font_color, font_thickness)
        #             concat = img
        #         elif pic_names[ranks[j-1]][:3]==pic_names[i][:3]:
        #             img = cv2.rectangle(img,(0,0),(w,h),(0,255,0),2) # the right pic
        #             img = cv2.putText(img, pic_names[ranks[j-1]][:3], text_pos, font, font_size, font_color, font_thickness)
        #             concat = cv2.hconcat([concat,img])
        #         else:
        #             img = cv2.rectangle(img,(0,0),(w,h),(0,0,255),2) # the wrong pic
        #             img = cv2.putText(img, pic_names[ranks[j-1]][:3], text_pos, font, font_size, font_color, font_thickness)
        #             concat = cv2.hconcat([concat,img])

        #     cv2.imwrite(res_dir+'res_'+pic_names[i],concat)


    print("done")


if __name__ == '__main__':
    line = "--config-file /home/fast-reid/configs/VeRi/sbs_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/veri_sbs_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    line1 = "--config-file /home/fast-reid/configs/VERIWild/bagtricks_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/veriwild_bot_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    line2 = "--config-file /home/fast-reid/configs/VehicleID/bagtricks_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/vehicleid_bot_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    args = default_argument_parser().parse_args(line)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
