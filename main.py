#!/usr/bin/env python
# coding: utf-8

import argparse
from argparse import Namespace

import os
import torch
import numpy as np

import torchvision.transforms as transforms

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy 
from pytorch_lightning.strategies.ddp import DDPStrategy

from solo.methods import BarlowTwins, SimCLR, MoCoV2Plus, MAE, MoCoV3, SimSiam, BYOL # imports the method class
from solo.utils.checkpointer import Checkpointer

from utils.custom_dataset import FilteredCIFAR10Dataset
from utils.supervised import SupervisedMethod
from utils.utils import load_json

from solo.data.pretrain_dataloader import (
    prepare_n_crop_transform,
    build_transform_pipeline,
)

from omegaconf import OmegaConf

def main():
    parser = argparse.ArgumentParser(description='Parameters for complexity project')
    # Required parameters
    parser.add_argument('--train_cfg', type=str, required=True,
                            help='set cfg file for training (SSL or FT)')                         
    parser.add_argument('--aug_stack_cfg', type=str, required=True,
                            help='set cfg file for augmentations')   
    parser.add_argument('--filter_cfg', type=str, required=True,
                            help='set cfg file for data filters')                       
    parser.add_argument('--nodes', type=int, required=True,
                            help='number of nodes') 
    parser.add_argument('--gpus', type=int, required=True,
                            help='number of gpus per node') 
     
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                            help='random seed (default: 0)')
    parser.add_argument('--method', type=str, default=None,
                            help='set method for pre-training')
    parser.add_argument('--name', type=str, default='self-supervised_learning',
                            help='set name for the run')
    parser.add_argument('--filter_train', type=bool, default=False,
                            help='set flag for filtering training data')
    parser.add_argument('--filter_test', type=bool, default=False,
                            help='set flag for filtering test data')
    parser.add_argument('--ckpt_path', type=str, default=None,
                            help='set path to load pre-trained model (requires backbone argument)')
    parser.add_argument('--windows', type=bool, default=False,
                        help='set backend to gloo if running on Windows')

    args = parser.parse_args()
    lin_probe = args.method is None
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_cfg_name = args.train_cfg
    kwargs = load_json(train_cfg_name, "models")
    cfg = OmegaConf.create(kwargs)

    methods_dict = {
        "barlow_twins": BarlowTwins,
        "simclr": SimCLR,
        "mocov2+": MoCoV2Plus,
        "mae": MAE,
        "mocov3": MoCoV3,
        "simsiam": SimSiam,
        "byol": BYOL
    }

    aug_stack_name = args.aug_stack_cfg
    transform_kwargs = load_json(aug_stack_name, "aug_stacks")
    default_format = transforms.Compose([ 
        transforms.Resize(transform_kwargs["crop_size"]),
        transforms.ToTensor(),               
    ])

    # if we are conducting a linear probe, use custom supervised method, otherwise use SSL method
    if lin_probe:
        model = SupervisedMethod(args.ckpt_path, cfg)
        transform = default_format
    else:
        model = methods_dict[args.method](cfg)
        transform_cfg = OmegaConf.create(transform_kwargs)
        transform = build_transform_pipeline("cifar10", transform_cfg)
        transform = prepare_n_crop_transform([transform], num_crops_per_aug=[kwargs["data"]["num_large_crops"]])

    filter_name = args.filter_cfg
    filter_data_dict = load_json(filter_name, "filters")

    # if we are looking to filter the training data to a subset of classes, then pass on the filtering information
    if args.filter_train:
        train_dataset = FilteredCIFAR10Dataset(classes=filter_data_dict, transform=transform)
    else:
        train_dataset = FilteredCIFAR10Dataset(transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=kwargs["optimizer"]["batch_size"], 
        num_workers=kwargs["num_workers"], 
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )

    if lin_probe:
        if args.filter_test:
            test_dataset = FilteredCIFAR10Dataset(classes=filter_data_dict, transform=default_format, train=False)
        else:
            test_dataset = FilteredCIFAR10Dataset(transform=default_format, train=False)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=kwargs["optimizer"]["batch_size"], 
            num_workers=kwargs["num_workers"], 
            pin_memory=True
        )

    wandb_logger = WandbLogger(
        name=args.name,  # name of the experiment
        project="complexity",  # name of the wandb project
        entity=None,
        offline=False,
        log_model=False,
        save_dir="../.."
    )
    wandb_logger.watch(model, log="gradients", log_freq=100)

    callbacks = []

    # automatically log our learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    os.environ['WANDB_NOTEBOOK_NAME'] = "complexity_project"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    if args.windows: os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

    run_name = args.name
    ckpt_args = OmegaConf.create({"name": run_name})

    # saves the checkpoint after every epoch
    ckpt = Checkpointer(
        ckpt_args,
        logdir="../../checkpoints/"+run_name,
        frequency=1,
    )
    callbacks.append(ckpt)

    if lin_probe: callbacks = None

    trainer_args = Namespace(**kwargs)

    trainer = Trainer.from_argparse_args(
        trainer_args,
        logger=wandb_logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator="cuda",
        num_nodes=args.nodes,
        devices=args.gpus,
        limit_val_batches=0,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader)

    if lin_probe:
        print("Downstream task:", args.filter_cfg)
        trainer.test(model, test_loader)

if __name__ == "__main__":
    main()