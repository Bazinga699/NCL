import torch
import torch.nn as nn
from backbone import res32_cifar, res50, res152
from modules import GAP, FCNorm, Identity

import numpy as np
import cv2
import os
import copy
import math
from torch.nn.parameter import Parameter
from net.MOCO import MoCo


class Cos_Classifier(nn.Module):
    """ plain cosine classifier """

    def __init__(self, num_classes=10, in_dim=640, scale=16, bias=False):
        super(Cos_Classifier, self).__init__()
        self.scale = scale
        self.weight = Parameter(torch.Tensor(num_classes, in_dim).cuda())
        self.bias = Parameter(torch.Tensor(num_classes).cuda(), requires_grad=bias)
        self.init_weights()

    def init_weights(self):
        self.bias.data.fill_(0.)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, **kwargs):
        ex = x / torch.norm(x.clone(), 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        out = torch.mm(ex, self.scale * ew.t()) + self.bias
        return out

class multi_Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000, use_dropout=False):
        super(multi_Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
               and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg
        self.network_num = len(self.cfg.BACKBONE.MULTI_NETWORK_TYPE)
        self.use_dropout = use_dropout

        if pretrain:

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                    pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL
                ) for i in range(self.network_num))
        else:

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                ) for i in range(self.network_num))

        self.module = nn.ModuleList(
            self._get_module()
            for i in range(self.network_num))

        if self.use_dropout:
            self.dropout = nn.ModuleList(
                nn.Dropout(p=0.5)
                for i in range(self.network_num))

        self.classifier = nn.ModuleList(
            self._get_multi_classifer(cfg.CLASSIFIER.BIAS, cfg.CLASSIFIER.TYPE)
            for i in range(self.network_num))

    def forward(self, input, **kwargs):

        if "feature_flag" in kwargs:
            return self.extract_feature(input, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.get_logits(input, **kwargs)

        logits = []
        for i in self.network_num:
            x = (self.backbone[i])(input[i], **kwargs)
            x = (self.module[i])(x)
            x = x.view(x.shape[0], -1)
            self.feat.append(copy.deepcopy(x))
            if self.use_dropout:
                x = (self.dropout[i])(x)
            x = (self.classifier[i])(x)
            logits.append(x)

        return logits

    def extract_feature(self, input, **kwargs):

        feature = []
        for i in range(self.network_num):
            x = (self.backbone[i])(input[i])
            x = (self.module[i])(x)
            x = x.view(x.shape[0], -1)
            feature.append(x)

        return feature

    def get_logits(self, input, **kwargs):

        logits = []
        for i in range(self.network_num):
            x = input[i]
            if self.use_dropout:
                x = (self.dropout[i])(x)
            x = (self.classifier[i])(x)
            logits.append(x)

        return logits

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_multi_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")

    def load_model(self, model_path, **kwargs):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if 'backbone_only' in kwargs.keys() and 'classifier' in k:
                continue;
            if k.startswith("module"):
                if k[7:] not in model_dict.keys():
                    print('not load:{}'.format(k))
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features

    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module = Identity()
        else:
            raise NotImplementedError

        return module

    def _get_multi_classifer(self, bias_flag, type):

        num_features = self.get_feature_length()
        if type == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif type == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif type == 'cos':
            classifier = Cos_Classifier(self.num_classes, num_features, scale=self.cfg.CLASSIFIER.COS_SCALE, bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier

class multi_Network_MOCO(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000, use_dropout=False):
        super(multi_Network_MOCO, self).__init__()
        pretrain = (
            True
            if mode == "train"
               and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.mlp_dim = cfg.NETWORK.MOCO_DIM
        self.num_classes = num_classes
        self.cfg = cfg
        self.network_num = len(self.cfg.BACKBONE.MULTI_NETWORK_TYPE)
        self.use_dropout = use_dropout

        if self.cfg.NETWORK.MOCO:
            self.MOCO = nn.ModuleList(
                MoCo(dim=cfg.NETWORK.MOCO_DIM, K=cfg.NETWORK.MOCO_K, T=cfg.NETWORK.MOCO_T)
                for i in range(self.network_num))

        if pretrain:

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                    pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL
                ) for i in range(self.network_num))
        else:

            self.backbone = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                ) for i in range(self.network_num))

        self.module = nn.ModuleList(
            self._get_module()
            for i in range(self.network_num))

        if self.use_dropout:
            self.dropout = nn.ModuleList(
                nn.Dropout(p=0.5)
                for i in range(self.network_num))

        self.classifier = nn.ModuleList(
            self._get_multi_classifer(cfg.CLASSIFIER.BIAS, cfg.CLASSIFIER.SEMI_TYPE)
            for i in range(self.network_num))
        self.feat = []

        if pretrain:
            self.backbone_MA = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                    pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL
                ) for i in range(self.network_num))
        else:
            self.backbone_MA = nn.ModuleList(
                eval(self.cfg.BACKBONE.MULTI_NETWORK_TYPE[i])(
                    self.cfg,
                    last_layer_stride=2,
                ) for i in range(self.network_num))

        for i in range(self.network_num):
            for param in self.backbone_MA[i].parameters():
                param.detach_()

        self.module_MA = nn.ModuleList(
            self._get_module()
            for i in range(self.network_num))
        for i in range(self.network_num):
            for param in self.module_MA[i].parameters():
                param.detach_()

        if self.use_dropout:
            self.dropout_MA = nn.ModuleList(
                nn.Dropout(p=0.5)
                for i in range(self.network_num))
            for i in range(self.network_num):
                for param in self.dropout_MA[i].parameters():
                    param.detach_()

        self.classifier_MA = nn.ModuleList(
            self._get_multi_classifer(cfg.CLASSIFIER.BIAS, cfg.CLASSIFIER.SEMI_TYPE)
            for i in range(self.network_num))
        for i in range(self.network_num):
            for param in self.classifier_MA[i].parameters():
                param.detach_()
        self.feat_MA = []

        if cfg.CLASSIFIER.TYPE == 'FC':
            self.classifier_ce = nn.ModuleList(
                nn.Linear(self.get_feature_length(), self.num_classes, cfg.CLASSIFIER.BIAS)
                for i in range(self.network_num))
        elif cfg.CLASSIFIER.TYPE == 'cos':
            self.classifier_ce = nn.ModuleList(
                Cos_Classifier(self.num_classes, in_dim=self.get_feature_length(), scale=cfg.CLASSIFIER.COS_SCALE, bias=True)
                for i in range(self.network_num))

    def forward(self, input, **kwargs):


        if "feature_flag" in kwargs:
            return self.extract_feature(input, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.get_logits(input, **kwargs)

        logits = []
        logits_ce = []
        for i in self.network_num:
            x = (self.backbone[i])(input[i], **kwargs)
            x = (self.module[i])(x)
            feature = x.view(x.shape[0], -1)
            self.feat.append(copy.deepcopy(feature))
            if self.use_dropout:
                feature = (self.dropout[i])(feature)

            output = (self.classifier[i])(feature)
            logits.append(output)

            output_ce = (self.classifier_ce[i])(feature)
            logits_ce.append(output_ce)

        logits_MA = []
        for i in self.network_num:
            x = (self.backbone_MA[i])(input[i], **kwargs)
            x = (self.module_MA[i])(x)
            x = x.view(x.shape[0], -1)
            self.feat_MA.append(copy.deepcopy(x))
            if self.use_dropout:
                x = (self.dropout_MA[i])(x)
            x = (self.classifier_MA[i])(x)
            logits_MA.append(x)

        return logits_ce, logits, logits_MA

    def extract_feature(self, input_all, **kwargs):

        input, input_MA = input_all

        feature = []
        for i in range(self.network_num):
            x = (self.backbone[i])(input[i], label=kwargs['label'][i])
            x = (self.module[i])(x)
            x = x.view(x.shape[0], -1)
            feature.append(x)

        feature_MA = []
        for i in range(self.network_num):
            x = (self.backbone_MA[i])(input_MA[i], label=kwargs['label'][i])
            x = (self.module_MA[i])(x)
            x = x.view(x.shape[0], -1)
            feature_MA.append(x)
        return feature, feature_MA

    def get_logits(self, input_all, **kwargs):

        input, input_MA = input_all
        logits = []
        logits_ce = []
        for i in range(self.network_num):
            feature = input[i]
            if self.use_dropout:
                feature = (self.dropout[i])(feature)

            output = (self.classifier[i])(feature)
            logits.append(output)

            output_ce = (self.classifier_ce[i])(feature)
            logits_ce.append(output_ce)

        logits_MA = []
        for i in range(self.network_num):
            x = input_MA[i]
            if self.use_dropout:
                x = (self.dropout_MA[i])(x)
            x = (self.classifier_MA[i])(x)
            logits_MA.append(x)

        return logits_ce, logits, logits_MA

    def extract_feature_maps(self, x):
        x = self.backbone(x)
        return x

    def freeze_multi_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone model has been loaded...")

    def load_model(self, model_path, **kwargs):
        pretrain_dict = torch.load(
            model_path, map_location="cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if 'backbone_only' in kwargs.keys() and 'classifier' in k:
                continue;
            if k.startswith("module"):
                if k[7:] not in model_dict.keys():
                    print('not load:{}'.format(k))
                    continue
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("All model has been loaded...")

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif 'res10' in self.cfg.BACKBONE.TYPE:
            num_features = 512
        else:
            num_features = 2048
        return num_features

    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module = Identity()
        else:
            raise NotImplementedError

        return module

    def _get_multi_classifer(self, bias_flag, type):

        num_features = self.get_feature_length()
        if type == "FCNorm":
            classifier = FCNorm(num_features, self.mlp_dim)
        elif type == "FC":
            classifier = nn.Linear(num_features, self.mlp_dim, bias=bias_flag)
        elif type == "mlp":
            classifier = nn.Sequential(nn.Linear(num_features, num_features, bias=bias_flag), \
                                       nn.ReLU(), \
                                       nn.Linear(num_features, self.mlp_dim, bias=bias_flag))
        elif type == 'cos':
            classifier = Cos_Classifier(self.mlp_dim, num_features, scale=self.cfg.CLASSIFIER.COS_SCALE, bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier
