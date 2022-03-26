from utils.registry import Registry
import torchvision.transforms as transforms
import random

TRANSFORMS = Registry()


@TRANSFORMS.register("random_resized_crop")
def random_resized_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomResizedCrop(
        size=size,
        scale=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.SCALE,
        ratio=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.RATIO,
    )


@TRANSFORMS.register("random_crop")
def random_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomCrop(
        size, padding=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_CROP.PADDING, fill=random.randint(0,255)
    )


@TRANSFORMS.register("random_horizontal_flip")
def random_horizontal_flip(cfg, **kwargs):
    return transforms.RandomHorizontalFlip(p=0.5)

@TRANSFORMS.register("random_vertical_flip")
def random_vertical_flip(cfg, **kwargs):
    return transforms.RandomVerticalFlip(p=0.5)

@TRANSFORMS.register("random_rotation20")
def random_rotation(cfg, **kwargs):
    return transforms.RandomRotation(degrees=20)

@TRANSFORMS.register("random_rotation10")
def random_rotation(cfg, **kwargs):
    return transforms.RandomRotation(degrees=10)

@TRANSFORMS.register("random_rotation30")
def random_rotation(cfg, **kwargs):
    return transforms.RandomRotation(degrees=30)


@TRANSFORMS.register("shorter_resize_for_crop")
def shorter_resize_for_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    assert size[0] == size[1], "this img-process only process square-image"
    return transforms.Resize(int(size[0] / 0.875))


@TRANSFORMS.register("normal_resize")
def normal_resize(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.Resize(size)


@TRANSFORMS.register("center_crop")
def center_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.CenterCrop(size)

@TRANSFORMS.register("normalize")
def normalize(cfg, **kwargs):
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

@TRANSFORMS.register("color_jitter")
def color_jitter(cfg, **kwargs):
    return transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0
    )