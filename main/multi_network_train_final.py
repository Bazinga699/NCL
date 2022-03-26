import time

from torch.nn import init

import _init_paths

from loss import *
from dataset import *
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_multi_model_final,
    get_category_list,
)

from core.function import multi_networks_train_model, multi_network_valid_model_final
from core.combiner import Combiner

import torch
import os
import shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast
from datetime import datetime

import random
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="code for NCL")
    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="/home/lijun/papers/NCL/config/CIFAR/CIFAR100/cifar100_im100_NCL_with_contrastive.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= False,
    )

    parser.add_argument(
        "--local_rank",
        help='local_rank for distributed training',
        type=int,
        default=0,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES']='4'
    if torch.cuda.is_available():
        print('using GPUS:%d' % torch.cuda.device_count())
    else:
        print('no GPU')

    args = parse_args()
    local_rank = args.local_rank
    rank = local_rank
    update_config(cfg, args)

    logger, log_file = create_logger(cfg, local_rank)
    warnings.filterwarnings("ignore")
    auto_resume = args.auto_resume

    setup_seed(cfg.RAND_SEED)

    # create model&log saving path
    if args.model_dir == None:
        model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models",
                                 str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    else:
        model_dir = args.model_dir
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes",
                             str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard",
                             str(datetime.now().strftime("%Y-%m-%d-%H-%M")))
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if local_rank == 0:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            logger.info(
                "This directory has already existed, Please remember to modify your cfg.NAME"
            )
            if not click.confirm(
                "\033[1;31;40mContinue and override the former directory?\033[0m",
                default=False,
            ):
                exit(0)
            if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
                shutil.rmtree(tensorboard_dir)
        print("=> output model will be saved in {}".format(model_dir))
        this_dir = os.path.dirname(__file__)
        ignore = shutil.ignore_patterns(
            "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
        )

    #DDP init
    if cfg.TRAIN.DISTRIBUTED:
        if local_rank == 0:
            print('Init the process group for distributed training')
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        if local_rank == 0:
            print('Init complete')

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)

    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    device = torch.device("cuda")

    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }

    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)

    epoch_number = cfg.TRAIN.MAX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    model = get_multi_model_final(cfg, num_classes, num_class_list, device, logger)
    combiner = Combiner(cfg, device, num_class_list)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    if cfg.TRAIN.DISTRIBUTED:
        model = model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = model.cuda()
        model = torch.nn.DataParallel(model)
    # ----- END MODEL BUILDER -----

    if cfg.TRAIN.DISTRIBUTED:
        train_sampler = torch.utils.data.DistributedSampler(train_set)

        trainLoader = DataLoader(
            train_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        validLoader = DataLoader(
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=True
        )

    else:

        trainLoader = DataLoader(
            train_set,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=cfg.TRAIN.SHUFFLE,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            drop_last=True
        )
        validLoader = DataLoader(
            valid_set,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TEST.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )

    if tensorboard_dir is not None and local_rank == 0:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1
    best_result_single, best_epoch_single = 0, 0

    # # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.TRAIN.DISTRIBUTED else "cuda"
        )
        model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']
    # ----- END RESUME ---------

    if rank == 0:
        logger.info(
            "-------------------Train start :{}  {}  {}-------------------".format(
                cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
            )
        )


    for epoch in tqdm(range(start_epoch, epoch_number + 1)):

        if cfg.TRAIN.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        scheduler.step()
        train_acc, train_loss = multi_networks_train_model(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            combiner,
            criterion,
            cfg,
            logger,
            writer=writer,
            rank=local_rank,
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}.pth".format(epoch),
        )
        if epoch % cfg.SAVE_STEP == 0 and local_rank == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc, valid_loss, valid_acc_single = multi_network_valid_model_final(
                validLoader, epoch, model, cfg, criterion, logger, device,
                rank=rank, distributed=cfg.TRAIN.DISTRIBUTED, writer=writer
            )
            loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
            if epoch % cfg.SAVE_STEP == 0 and local_rank == 0:
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, model_save_path)
            if valid_acc > best_result and local_rank == 0:
                best_result, best_epoch = valid_acc, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "best_ensemble_model.pth")
                )
            if valid_acc_single > best_result_single and local_rank == 0:
                best_result_single, best_epoch_single = valid_acc_single, epoch
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_result': best_result,
                    'best_epoch': best_epoch,
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "best_single_model.pth")
                )
            if rank == 0:
                logger.info(
                    "--------------Best_ensemble_Epoch:{:>3d}    Best_ensemble_Acc:{:>5.2f}%--------------".format(
                        best_epoch, best_result * 100
                    )
                )
                logger.info(
                    "--------------Best_single_Epoch:{:>3d}    Best_single_Acc:{:>5.2f}%--------------".format(
                        best_epoch_single, best_result_single * 100
                    )
                )

        if cfg.TRAIN.TENSORBOARD.ENABLE and local_rank == 0:
            writer.add_scalars("scalar/acc", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE and local_rank == 0:
        writer.close()
    if rank == 0:
        logger.info(
            "-------------------Train Finished :{}~{}~{}~-------------------".format(cfg.NAME, best_result*100, best_result_single*100)
        )
