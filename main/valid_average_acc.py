import _init_paths

from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_multi_model_final,
    get_category_list,
)
from net import  multi_Network, multi_Network_MOCO

import copy
def parse_args():
    parser = argparse.ArgumentParser(description="evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="/home/lijun/papers/Long_Tailed/configs/cifar100_im100_NCL.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def my_shot_acc(predict, label, many_shot_thr, low_shot_thr, train_class_dict):

    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    class_num = len(train_class_dict)
    class_correct = np.zeros(class_num)
    train_class_sum = np.zeros(class_num)
    test_class_sum = np.zeros(class_num)
    for i in range(class_num):
        class_correct[i] = (predict[label == i] == label[label == i]).sum()
        train_class_sum[i] = len(train_class_dict[i])
        test_class_sum[i] = len(label[label == i])

    many_shot_correct = 0
    many_shot_all = 0

    median_shot_correct = 0
    median_shot_all = 0

    few_shot_correct = 0
    few_shot_all = 0

    for i in range(class_num):
        if train_class_sum[i] >= many_shot_thr:
            many_shot_correct += class_correct[i]
            many_shot_all += test_class_sum[i]
        elif train_class_sum[i] <= low_shot_thr:
            few_shot_correct += class_correct[i]
            few_shot_all += test_class_sum[i]
        else:
            median_shot_correct += class_correct[i]
            median_shot_all += test_class_sum[i]

    print('{:>5.2f}\t{:>5.2f}\t{:>5.2f}\t{:>5.2f}'.format(many_shot_correct/many_shot_all*100, \
                                                       median_shot_correct/median_shot_all*100, \
                                                       few_shot_correct/few_shot_all*100, \
                                                       (many_shot_correct+median_shot_correct+few_shot_correct)/(many_shot_all+median_shot_all+few_shot_all)*100))

def valid_model(dataLoader, model, cfg, device, train_class_dict):
    model.eval()
    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
    every_network_predict = [[] for _ in range(network_num)]
    every_network_logits = [[] for _ in range(network_num)]
    every_network_feature = [[] for _ in range(network_num)]

    average_predict = []

    all_label = []
    with torch.no_grad():

        for i, (image, label, meta) in tqdm(enumerate(dataLoader)):

            image, label = image.to(device), label.to(device)
            image_list = [image for i in range(network_num)]
            all_label.append(label.cpu())

            if cfg.NETWORK.MOCO:
                feature = model((image_list, image_list), label=label, feature_flag=True)

                for i in range(network_num):
                    every_network_feature[i].append(feature[0][i])

                output_ce, output, output_MA = model(feature, classifier_flag=True)
            else:
                feature = model(image_list, label=label, feature_flag=True)

                for i in range(network_num):
                    every_network_feature[i].append(feature[i])

                output_ce = model(feature, classifier_flag=True)

            sum_result = copy.deepcopy(output_ce[0])
            for i in range(network_num):
                if i > 0:
                    sum_result += output_ce[i]
            average_predict.append(sum_result.argmax(dim=1).cpu())



            for j, logit in enumerate(output_ce):
                every_network_logits[j].append(logit)
                every_network_predict[j].append(torch.argmax(logit, dim=1).cpu())


    all_label = torch.cat(all_label)

    average_predict = torch.cat(average_predict)


    average_acc = torch.sum(average_predict==all_label) / all_label.shape[0]
    print('average_acc: {}'.format(average_acc))



    print('average')
    my_shot_acc(average_predict, all_label, 100, 20, train_class_dict)

    for i in range(network_num):
        every_network_predict[i] = torch.cat(every_network_predict[i])
        print('network {}'.format(i))
        my_shot_acc(every_network_predict[i], all_label, 100, 20, train_class_dict)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cuda")
    if cfg.NETWORK.MOCO:
        model = multi_Network_MOCO(cfg, mode="train", num_classes=num_classes, num_class_list=[],
                                   use_dropout=cfg.DROPOUT, MA_model=cfg.NETWORK.MA_MODEL)
    else:
        model = multi_Network(cfg, mode="train", num_classes=num_classes, use_dropout=cfg.DROPOUT,
                              MA_model=cfg.NETWORK.MA_MODEL)

    model_dir = '/media/ssd1/lijun/output/CVPR2022/test/models/2021-11-05-00-52/best_model.pth'

    model.load_model(model_dir)

    model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    valid_model(testLoader, model, cfg, device, train_set.class_dict)
