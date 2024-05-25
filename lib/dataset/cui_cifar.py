from dataset.baseset import BaseSet
import numpy as np
import random
from utils.utils import get_category_list
import math
import torchvision.transforms as transforms
from PIL import ImageFilter
from dataset.autoaug import CIFAR10Policy, Cutout
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def aug_plus(aug_comb='cifar100', mode='train', plus_plus='False'):
    # PaCo's aug: https://github.com/jiequancui/Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if plus_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    augmentation_regular = [
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    augmentation_sim_cifar = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]


    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]
    if aug_comb == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_comb == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_comb == 'cifar100':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation_sim_cifar)]

    if mode == 'train':
        return transform_train
    else:
        return val_transform

class MULTI_NETWORK_CIFAR_AUGPLIS(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()

        self.transform = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')


        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        if 'weighted' in self.sample_type \
                and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        if self.mode != 'train':
            image = self.transform(img)
        else:
            image = self.transform[0](img)
        meta = dict({'image_id': index})
        image_label = now_info['category_id']
        return image, image_label, meta

class MULTI_NETWORK_CIFAR_MOCO_AUGPLIS(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.network_num = len(cfg.BACKBONE.MULTI_NETWORK_TYPE)
        self.mode = mode

        # strong augmentation. remove it you will get weak augmentation which is defined in BaseSet.update_transform() .
        self.transform = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')

        self.class_dict = self._get_class_dict()
        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            #print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        if 'weighted' in self.sample_type \
                and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        meta = dict({'image_id': index})
        image_label = now_info['category_id']

        if self.mode != 'train':
            image1 = self.transform(img)
            return image1, image_label, meta

        image1 = self.transform[0](img)
        image2 = self.transform[1](img)

        return (image1, image2), image_label, meta


class MULTI_NETWORK_CIFAR_AUGPLIS_NCL_PLUS(BaseSet):
    def __init__(self, mode = 'train', cfg = None, sample_id = 0, transform = None):
        super().__init__(mode, cfg, transform)
        self.sample_id = sample_id
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()
        self.intra_N = cfg.DATASET.INTRA_N

        self.transform = aug_plus(aug_comb='cifar100', mode=mode, plus_plus='False')


        if mode == 'train':
            if 'weighted' in self.sample_type:
                self.class_weight, self.sum_weight = self.get_weight(self.data, self.num_classes)
                print('-' * 20 + ' dataset' + '-' * 20)
                print('multi_network: %d class_weight is (the first 10 classes): '%sample_id)
                print(self.class_weight[:10])

                num_list, cat_list = get_category_list(self.get_annotations(), self.num_classes, self.cfg)

                self.instance_p = np.array([num / sum(num_list) for num in num_list])
                self.class_p = np.array([1 / self.num_classes for _ in num_list])
                num_list = [math.sqrt(num) for num in num_list]

                self.square_p = np.array([pow(num, 0.5) / sum(pow(np.array(num_list), 0.5)) for num in num_list])

                self.class_dict = self._get_class_dict()

    def update(self, epoch):
        self.epoch = epoch
        if self.sample_type == "weighted_progressive":
            self.progress_p = epoch/self.cfg.TRAIN.MAX_EPOCH * self.class_p + (1-epoch/self.cfg.TRAIN.MAX_EPOCH)*self.instance_p
            print('self.progress_p', self.progress_p)

    def __getitem__(self, index):
        if 'weighted' in self.sample_type \
                and self.mode == 'train':
            assert self.sample_type in ["weighted_balance", 'weighted_square', 'weighted_progressive']
            if self.sample_type == "weighted_balance":
                sample_class = random.randint(0, self.num_classes - 1)
            elif self.sample_type == "weighted_square":
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.square_p)
            else:
                sample_class = np.random.choice(np.arange(self.num_classes), p=self.progress_p)
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
        now_info = self.data[index]
        img = self._get_image(now_info)
        if self.mode != 'train':
            image = self.transform(img)
            meta = dict({'image_id': index})
            image_label = now_info['category_id']  # 0-index
            return image, image_label, meta
        else:
            image = []
            image_label = []

            for i in range(self.intra_N):
                image.append(self.transform[0](img))
                image_label.append(now_info['category_id'])
                meta = dict({'image_id': index})
                
            return image, image_label, meta
