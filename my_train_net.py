#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from fastreid.utils.checkpoint import Checkpointer
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.config import get_cfg
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append('.')

from aicity2020model import aicity_2020_model

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.TEST.NORMAL = False
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

        model = aicity_2020_model()

        res = DefaultTrainer.test(cfg, model)
        return res

    #    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    # Note that it does not load any weights from ``cfg``.
    trainer = DefaultTrainer(cfg)

    Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS) 
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    home_dir = "/data/codes/fast-reid/"
    line = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/veri_sbs_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    
    line1 = f"--config-file {home_dir}configs/VehicleID/bagtricks_R50-ibn_veri.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/vehicleid_bot_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    
    line2 = f"--config-file {home_dir}configs/VehicleID/bagtricks_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/vehicleid_bot_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()

    line3 = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn_vehicleID.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/veri_sbs_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    
    line4 = f"--config-file {home_dir}configs/VERIWild/bagtricks_R50-ibn_VehicleID.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/veriwild_bot_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    
    line5 = f"--config-file {home_dir}configs/VERIWild/bagtricks_R50-ibn_veri.yml \
        --eval-only MODEL.WEIGHTS /data/data/fastreid_pth/veriwild_bot_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()

    # train
    line6 = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn.yml \
         --resume \
         MODEL.WEIGHTS /data/data/fastreid_pth/veri_sbs_R50-ibn.pth \
        MODEL.DEVICE \"cuda:0\" ".split()

    line7 = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn.yml \
        --resume \
         MODEL.WEIGHTS {home_dir}logs/veri/sbs_R50-ibn/model_best.pth \
        MODEL.DEVICE \"cuda:0\" ".split()
    
    line8 = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn.yml \
        --eval-only MODEL.WEIGHTS {home_dir}logs/veri/sbs_R50-ibn/model_best.pth \
        MODEL.DEVICE \"cuda:0\" ".split()

    line9 = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn.yml \
        MODEL.DEVICE \"cuda:0\" ".split()

    # test other model, pay attention to input size
    line10 = f"--config-file {home_dir}configs/VeRi/sbs_R50-ibn_aicity2020.yml \
        --eval-only MODEL.DEVICE \"cuda:0\"".split()

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
