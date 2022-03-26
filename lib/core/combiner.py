
import torch, math
from core.evaluate import accuracy
from torch.nn import functional as F
from net.MOCO import shuffle_BN, shuffle_BN_DDP, unshuffle_BN_DDP, unshuffle_BN

class Combiner:
    def __init__(self, cfg, device, num_class_list=None):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.num_class_list = torch.FloatTensor(num_class_list)
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.initilize_all_parameters()

    def initilize_all_parameters(self):

        self.CON_ratio = self.cfg.LOSS.CON_RATIO
        self.show_step = self.cfg.SHOW_STEP
        self.distributed = self.cfg.TRAIN.DISTRIBUTED

        print('_'*100)
        print('combiner type: ', self.type)
        print('_'*100)

    def update(self, epoch):
        self.epoch = epoch


    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )


    def multi_network_default(self, model, criterion, image, label, meta, **kwargs):

        for i in range(len(image)):
            image[i], label[i] = image[i].to(self.device), label[i].to(self.device)

        feature = model(image, feature_flag=True, label=label)
        output = model(feature, classifier_flag=True)

        loss = criterion(output, label)

        average_result = torch.mean(torch.stack(output), dim=0)
        now_result = torch.argmax(average_result, 1)
        now_acc = accuracy(now_result.cpu().numpy(), label[0].cpu().numpy())[0]

        return loss, now_acc

    def multi_network_default_CON(self, model, criterion, image, label, meta, **kwargs):

        image_p = []
        image_k = []
        for i in range(len(image)):
            image_p.append(image[i][0].to(self.device))
            image_k.append(image[i][1].to(self.device))
            label[i] = label[i].to(self.device)

        # shuffle BN
        if self.distributed:
            image_k, idx_unshuffle = shuffle_BN_DDP(image_k)
            pass
        else:
            image_k, idx_unshuffle = shuffle_BN(image_k)

        feature = model((image_p, image_k), feature_flag=True, label=label)
        output_ce, output_p, output_k = model(feature, classifier_flag=True)

        # unshuffle
        if self.distributed:
            output_k = unshuffle_BN_DDP(output_k, idx_unshuffle)
        else:
            output_k = unshuffle_BN(output_k, idx_unshuffle)

        loss_ce = criterion(output_ce, label, feature=feature, classifier=model.module.classifier)

        average_result = torch.mean(torch.stack(output_ce), dim=0)
        now_result = torch.argmax(average_result, 1)
        now_acc = accuracy(now_result.cpu().numpy(), label[0].cpu().numpy())[0]

        # contrastive_loss
        loss_CON = 0
        for i, (q, k) in enumerate(zip(output_p, output_k)):
            q = F.normalize(q, dim=1)
            k = F.normalize(k, dim=1)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, model.module.MOCO[i].queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= model.module.MOCO[i].T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            if self.distributed:
                model.module.MOCO[i]._dequeue_and_enqueue_DDP(k)
            else:
                model.module.MOCO[i]._dequeue_and_enqueue(k)


            loss_CON += F.cross_entropy(logits, labels)

        loss = loss_ce + loss_CON * self.CON_ratio

        return loss, now_acc
