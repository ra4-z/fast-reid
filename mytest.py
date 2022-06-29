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
            transforms.Resize((256, 256), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mydataset = MyDataset(transform=data_transforms)
        dataloader = DataLoader(mydataset, batch_size=16)

        features, camids, labels, pic_names = extract_info(model, dataloader)

        query_feature = gallery_feature = features
        query_feature = torch.FloatTensor(query_feature).cuda()
        gallery_feature = torch.FloatTensor(gallery_feature).cuda()
        dist_mat = compute_distance_matrix(query_feature, gallery_feature)
        dist_mat = dist_mat.cpu().numpy()

        import os
        k = 10
        topk = np.argsort(dist_mat, axis=1)[:, :k]
        res = []
        res_root = '/home/fast-reid/res20220628/'
        src_root = '/home/data/frames20220628/'
        if not os.path.exists(res_root):
            os.makedirs(res_root)
        for i, line in enumerate(topk):
            src_pic = src_root+pic_names[line[0]]
            res_pic_path = res_root+pic_names[line[0]][:-5]
            if not os.path.exists(res_pic_path):
                os.makedirs(res_pic_path)
            os.system(f'cp {src_pic} {res_pic_path}/src.jpg')

            for j, idx in enumerate(line):
                res_pic = src_root + pic_names[idx]
                rest_res_pic_path = res_pic_path+'/'+str(j)+"_"+pic_names[idx]
                os.system(f'cp {res_pic} {rest_res_pic_path}')




        all_cmc, all_AP, all_INP = evaluate_rank(
            dist_mat, labels, labels, camids, camids, 10)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        print(
            f"mAP:{mAP}\t R1:{all_cmc[0]}\t R5:{all_cmc[4]}\t R10:{all_cmc[9]}")

    print("done")


if __name__ == '__main__':
    line = "--config-file /home/fast-reid/configs/VeRi/sbs_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS /home/data/feat/feat/veri_sbs_R50-ibn.pth \
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
