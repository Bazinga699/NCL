
from core.evaluate import accuracy, AverageMeter
import torch
import time

def multi_networks_train_model(
        trainLoader, model, epoch, epoch_number, optimizer, combiner, criterion, cfg, logger, rank=0, **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
    trainLoader.dataset.update(epoch)
    combiner.update(epoch)
    criterion.update(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):

        image_list = [image] * network_num
        label_list = [label] * network_num
        meta_list = [meta] * network_num

        if isinstance(label, list):
            cnt = label[0].shape[0]
        else:
            cnt = label.shape[0]

        optimizer.zero_grad()

        loss, now_acc = combiner.forward(model, criterion, image_list, label_list, meta_list, now_epoch=epoch,
                                         train=True, cfg=cfg, iteration=i, log=logger,
                                         class_list=criterion.num_class_list)


        if cfg.NETWORK.MOCO:
            alpha = cfg.NETWORK.MA_MODEL_ALPHA
            for net_id in range(network_num):
                net = ['backbone', 'module']
                for name in net:
                    for ema_param, param in zip(eval('model.module.' + name + '_MA').parameters(),
                                                eval('model.module.' + name).parameters()):
                        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


        loss.backward()
        optimizer.step()
        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)
    return acc.avg, all_loss.avg

def multi_network_valid_model_final(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, **kwargs
):
    model.eval()
    network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
    cnt_all = 0
    every_network_result = [0 for _ in range(network_num)]

    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()

        for i, (image, label, meta) in enumerate(dataLoader):

            image, label = image.to(device), label.to(device)
            image_list = [image for i in range(network_num)]

            if cfg.NETWORK.MOCO:
                feature = model((image_list,image_list), label=label, feature_flag=True)
                output_ce, output, output_MA = model(feature, classifier_flag=True)
            else:
                feature = model(image_list, label=label, feature_flag=True)
                output_ce = model(feature, classifier_flag=True)

            #loss = criterion(output_ce, (label,))

            for j, logit in enumerate(output_ce):
                every_network_result[j] += torch.sum(torch.argmax(logit, dim=1).cpu() == label.cpu())

            average_result = torch.mean(torch.stack(output_ce), dim=0)
            now_result = torch.argmax(average_result, 1)

            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            cnt_all += cnt
            #all_loss.update(loss.data.item(), cnt)
            acc_avg.update(acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_ensemble_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100
        )
        if rank == 0:
            for i, result in enumerate(every_network_result):
                logger.info("network {} Valid_single_Acc: {:>5.2f}%".format(i, float(result) / cnt_all * 100))
            logger.info(pbar_str)
        best_single_acc = max(every_network_result) / cnt_all
    return acc_avg.avg, all_loss.avg, best_single_acc
