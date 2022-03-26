from dataset.baseset import BaseSet
import random, cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter
from dataset.randaugment import rand_augment_transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode='train', randaug_n=2, randaug_m=10, plus_plus='False'):
    # PaCo's aug: https://github.com/jiequancui/ Parametric-Contrastive-Learning

    normalize = transforms.Normalize(mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]) if dataset == 'inat' \
        else transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_sim = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation_sim02 = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    augmentation_randnclsstack = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    augmentation_randncls = [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(randaug_n, randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])

    if aug_type == 'regular_regular':
        transform_train = [transforms.Compose(augmentation_regular), transforms.Compose(augmentation)]
    elif aug_type == 'mocov2_mocov2':
        transform_train = [transforms.Compose(augmentation), transforms.Compose(augmentation)]
    elif aug_type == 'sim_sim':
        transform_train = [transforms.Compose(augmentation_sim), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randcls_sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim)]
    elif aug_type == 'randclsstack_sim02':
        transform_train = [transforms.Compose(augmentation_randnclsstack), transforms.Compose(augmentation_sim02)]

    if mode == 'train':
        return transform_train
    else:
        return val_transform


class iNaturalist(BaseSet):
    def __init__(self, mode='train', cfg=None, sample_id = 0, transform=None):
        super(iNaturalist, self).__init__(mode, cfg, transform)
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()
        self.transform = aug_plus(dataset='inat', aug_type='randcls_sim', mode=mode, plus_plus='False')

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
        meta = dict()
        image_label = now_info['category_id']  # 0-index
        return image, image_label, meta

class ImageNet_LT(BaseSet):
    def __init__(self, mode='train', cfg=None,sample_id = 0, transform=None):
        super(ImageNet_LT, self).__init__(mode, cfg, transform)
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()
        self.transform = aug_plus(dataset='ImageNet_LT', aug_type='randcls_sim', mode=mode, plus_plus='False')

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
        meta = dict()
        image_label = now_info['category_id']  # 0-index
        return image, image_label, meta

class Places_LT_MOCO(BaseSet):
    def __init__(self, mode='train', cfg=None,sample_id = 0, transform=None):
        super(Places_LT_MOCO, self).__init__(mode, cfg, transform)
        self.sample_type = cfg.TRAIN.SAMPLER.MULTI_NETWORK_TYPE[sample_id]
        self.class_dict = self._get_class_dict()
        self.transform = aug_plus(dataset='ImageNet_LT', aug_type='sim_sim', mode=mode, plus_plus='False')
        self.mode = mode

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
        meta = dict()
        image_label = now_info['category_id']  # 0-index

        if self.mode != 'train':
            image = self.transform(img)
            return image, image_label, meta

        image1 = self.transform[0](img)
        image2 = self.transform[1](img)
        return (image1, image2), image_label, meta







