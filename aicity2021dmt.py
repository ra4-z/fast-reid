import sys
sys.path.append("./AICITY2021_Track2_DMT/")
import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
from torchvision import datasets, models, transforms
from MyDataloader import MyDataset
from torch.utils.data import Dataset, DataLoader
from MyEval import extract_info, calculate_result


def aicity_2021_dmt_model():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/home/AICITY2021_Track2_DMT/configs/stage2/resnext101a_384.yml",
        help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, num_class=1)
    model = model.eval()
    model = model.cuda()

    model.load_param(cfg.TEST.WEIGHT)

    return model

if __name__ == "__main__":
    model = aicity_2021_dmt_model()