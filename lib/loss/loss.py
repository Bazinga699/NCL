import torch
import torch.nn as nn
from torch.nn import functional as F


def NBOD(inputs, factor):

    classifier_num = len(inputs)
    if classifier_num == 1:
        return 0
    logits_softmax = []
    logits_logsoftmax = []
    for i in range(classifier_num):
        logits_softmax.append(F.softmax(inputs[i], dim=1))
        logits_logsoftmax.append(torch.log(logits_softmax[i] + 1e-9))

    loss_mutual = 0
    for i in range(classifier_num):
        for j in range(classifier_num):
            if i == j:
                continue
            loss_mutual += factor * F.kl_div(logits_logsoftmax[i], logits_softmax[j],reduction='batchmean')
    loss_mutual /= (classifier_num - 1)
    return  loss_mutual

def NBOD_NCL_plus(inputs, factor, **kwargs):

    classifier_num = len(inputs) # classifier num in inter_CL, aug num in intra_CL 
    if classifier_num == 1:
        return 0
    logits_softmax = []
    logits_logsoftmax = []
    for i in range(classifier_num):
        logits_softmax.append(F.softmax(inputs[i], dim=1))
        logits_logsoftmax.append(torch.log(logits_softmax[i] + 1e-9))

    loss_mutual_all = 0
    for i in range(classifier_num):
        for j in range(classifier_num):
            if i == j:
                continue
            loss_mutual_all += F.kl_div(logits_logsoftmax[i], logits_softmax[j],reduction='batchmean')
    loss_mutual_all /= ((classifier_num - 1) * classifier_num / 2)
    loss_mutual_all *= factor

    mask = kwargs['MASK']

    loss_mutual_partial = 0
    for i in range(classifier_num):
        for j in range(classifier_num):
            if i >= j:
                continue

            loss_mutual_partial += F.kl_div(logits_logsoftmax[i] * mask, logits_softmax[j] * mask,
                                            reduction='batchmean')
            loss_mutual_partial += F.kl_div(logits_logsoftmax[j] * mask, logits_softmax[i] * mask,
                                            reduction='batchmean')

    loss_mutual_partial /= ((classifier_num - 1) * classifier_num / 2)
    loss_mutual_partial *= factor

    return  loss_mutual_all + loss_mutual_partial

class NIL_NBOD(nn.Module):
    def __init__(self, para_dict=None):
        super(NIL_NBOD, self).__init__()
        self.para_dict = para_dict
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

        self.multi_classifier_diversity_factor = self.para_dict['cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR
        self.multi_classifier_diversity_factor_hcm = self.para_dict['cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR_HCM
        self.hcm_N = self.para_dict['cfg'].LOSS.HCM_N
        self.hcm_ratio = self.para_dict['cfg'].LOSS.HCM_RATIO
        self.ce_ratio = self.para_dict['cfg'].LOSS.CE_RATIO

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_N]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets[0])

        loss += NBOD(inputs_balance, factor=self.multi_classifier_diversity_factor)
        loss += NBOD(inputs_HCM_balance, factor=self.multi_classifier_diversity_factor_hcm)
        loss += los_ce * self.ce_ratio + loss_HCM * self.hcm_ratio
        return loss

    def update(self, epoch):
        """
        Args:
           code can be added for progressive loss.
        """
        pass


class NIL_NBOD_plus(nn.Module):
    def __init__(self, para_dict=None):
        super(NIL_NBOD_plus, self).__init__()
        self.para_dict = para_dict
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']
        self.bsce_weight = torch.FloatTensor(self.num_class_list).to(self.device)

        self.multi_classifier_diversity_factor = self.para_dict['cfg'].LOSS.MULTI_CLASIIFIER_LOSS.DIVERSITY_FACTOR
        self.hcm_N = self.para_dict['cfg'].LOSS.HCM_N
        self.hcm_ratio = self.para_dict['cfg'].LOSS.HCM_RATIO
        self.ce_ratio = self.para_dict['cfg'].LOSS.CE_RATIO
        self.intra_N = self.para_dict['cfg'].DATASET.INTRA_N

    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (classifier_num, batch_size, num_classes)
            targets: ground truth labels with shape (classifier_num, batch_size)
        """
        classifier_num = len(inputs)
        loss_HCM = 0
        loss = 0
        los_ce = 0

        inputs_HCM_balance = []
        inputs_balance = []
        class_select = inputs[0].scatter(1, targets[0].unsqueeze(1), 999999)
        class_select_include_target = class_select.sort(descending=True, dim=1)[1][:, :self.hcm_N]
        mask = torch.zeros_like(inputs[0]).scatter(1, class_select_include_target, 1)
        for i in range(classifier_num):

            logits = inputs[i] + self.bsce_weight.unsqueeze(0).expand(inputs[i].shape[0], -1).log()
            inputs_balance.append(logits)
            inputs_HCM_balance.append(logits * mask)

            los_ce += F.cross_entropy(logits, targets[0])
            loss_HCM += F.cross_entropy(inputs_HCM_balance[i], targets[0])

        true_batch = int(targets[0].shape[0] / self.intra_N)
        inputs_balance_pice = [[] for i in range(classifier_num)]



        for net_i in range(classifier_num):
            begin_pos = 0
            end_pos = true_batch
            for i in range(self.intra_N):
                inputs_balance_pice[net_i].append(inputs_balance[net_i][begin_pos:end_pos])
                begin_pos += true_batch
                end_pos += true_batch

        loss_intra = 0
        for i in range(classifier_num):
            loss_intra += NBOD_NCL_plus(inputs_balance_pice[i], self.multi_classifier_diversity_factor, MASK=mask[:true_batch])


        loss_inter = NBOD_NCL_plus(inputs_balance, self.multi_classifier_diversity_factor, MASK=mask)

        loss += los_ce * self.ce_ratio + loss_HCM * self.hcm_ratio + loss_intra + loss_inter
        return loss

    def update(self, epoch):
        """
        Args:
           code can be added for progressive loss.
        """
        pass


if __name__ == '__main__':
    pass