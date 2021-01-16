# -*- coding: utf-8 -*-
"""
@author: Rajesh

"""
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from dataloader.ufs_data import UFSegmentationDataset, KittiDataset
from torch.utils.data import DataLoader
from dataloader.train_val_test_split import TrainValTestSplit, ExistingTrainValTestSplit
from PIL import Image
import torch

# gpu = 0
# if len(sys.argv) > 1:
#     gpu = sys.argv[1]
#
# # we have to do this before importing torch
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
# print(f"using gpu {os.environ['CUDA_VISIBLE_DEVICES']}")

import numpy as np
from pathlib import Path
from segmentation_models_pytorch import PSPNet
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from torchvision import transforms
from runtime.metrics import AreaCoveredMetric, AccuracyMetric, IOUMetric, MAEMetric, MSEMetric, \
    FalseNegativesMetric, TruePositiveMetric, FalseNegativeRateMetric, IOUMultiClassMetric
from runtime.training import ModelTrainer
from runtime.utils import parse_args, dice_loss, iou_loss, multiclass_dice_loss
from dataloader.preprocessing import ApplyPreprocessing
from runtime.evaluation import evaluate_segmentation


def make_segmentation_net(cfg, data_dir, generate_org=False):
    # MobilenetV2 encoder and PSPNET decoder for segmentation
    net = PSPNet(encoder_name="mobilenet_v2", in_channels=1, encoder_weights="imagenet", activation="sigmoid")
    # Pre-processing function for inputs
    preprocess_f = get_preprocessing_fn(encoder_name='mobilenet_v2', pretrained='imagenet')
    width = cfg['data']['width']
    height = cfg['data']['height']
    bbox_dir = cfg['data']['bbox_dir']
    if cfg['data']['rescale'] == "bbox":
        dataset = UFSegmentationDataset(data_root=data_dir, bbox_dir=bbox_dir, transform=transforms.Compose(
            [transforms.Resize(size=(height, width), interpolation=Image.NEAREST),transforms.ToTensor()]))
    else:
        dataset = UFSegmentationDataset(data_root=data_dir, bbox_dir=None, transform=transforms.Compose(
            [# ApplyPreprocessing(preprocess_f),
            transforms.Resize(size=(height, width), interpolation=Image.NEAREST),
            transforms.ToTensor()]))
    # KITTI dataset
    # dataset = KittiDataset(data_root=data_dir, mode="train", transform=transforms.Compose([preprocess_f, transforms.Resize(width, height), transforms.ToTensor()]))

    return net, dataset


def train_base(cfg, model, dataset, metrics, eval_f, loss_fn=MSELoss(), save_every_n_epochs=5, indx_dir="indices", test_split=0.1,
               shuffle_train_val=False):
    train, debug, train_on_gpu = parse_args()
    break_after_n_batches = 5
    epochs = cfg['train']['max_epochs']
    batch_size = cfg['train']['batch_size']
    use_gpu = True if torch.cuda.is_available() else False
    model_path = cfg['train']['saved_model_path']
    training = cfg['train']['training']

    split = TrainValTestSplit(indx_dir, dataset)

    print("training" if training else "evaluating")

    if not split.split_exists() and not train:
        raise RuntimeError("in eval mode, but cannot find data split")
    elif not split.split_exists():
        split.make_data_split(val_split=0.1, test_split=test_split, shuffle_train_val=shuffle_train_val)

    l_train, l_validation, l_test = split.make_dataloaders(batch_size, workers=0 if debug else 4)

    # train_loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=False)
    if training:
        trainer = ModelTrainer(loss_fn=loss_fn, use_gpu=use_gpu, model_path=model_path)
        for m in metrics:
            trainer.add_metric(m)

        trainer.set_model(model)
        trainer.train_model(l_train, l_validation, n_epochs=epochs, save_every_n_epochs=save_every_n_epochs)
    else:
        trainer = ModelTrainer(use_gpu=False, loss_fn=loss_fn, model_path=model_path)
        trainer.load()
        eval_f(trainer, l_test)
# Loading config file is taken care by hydra
@hydra.main(config_name="config")
def train_segmentation_net_ufs(cfg: DictConfig):
    # prints a styled output of the configuration details in the terminal
    print(OmegaConf.to_yaml(cfg))
    data_dir = cfg['data']['dir']
    model, dataset = make_segmentation_net(cfg, data_dir)

    # metrics = [AreaCoveredMetric(), IOUMetric(), AccuracyMetric(), FalseNegativeRateMetric(),
    #            FalseNegativesMetric(), TruePositiveMetric()]
    metrics = [IOUMetric(), AccuracyMetric()]
    idx_dir = cfg["data"]['idx_dir']
    split = TrainValTestSplit(idx_dir, dataset)

    eval_f = lambda trainer, data: evaluate_segmentation(cfg,
        trainer, dataset, split)
    train_base(cfg, model, dataset, metrics, eval_f, loss_fn=dice_loss, indx_dir=idx_dir, test_split=0.1,
               shuffle_train_val=True)

if __name__ == "__main__":
    train_segmentation_net_ufs()