import albumentations as aug
from typing import Tuple


def image_augmentation(config: list, target_shape: Tuple[int, int, int]) -> aug.core.composition.Compose:
    """
    Method for creating a list of transforms according to the configuration.
    :param config: list with a set of parameters for augmentation.
    :param target_shape: tuple with input shape of image.
    :return: composed transforms.
    """
    augmentation_list = []
    augmentations = {'vertical_flip': aug.VerticalFlip(),
                     'horizontal_flip': aug.HorizontalFlip(),
                     'sharpen': aug.Sharpen(alpha=(0.7, 1.0), lightness=(0.5, 0.7)),
                     'rgb_shift': aug.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
                     'brightness_contrast': aug.RandomBrightnessContrast(brightness_limit=0.1,
                                                                         contrast_limit=(0, 0.25)),
                     'hue_saturation': aug.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=30),
                     'distortion': aug.OneOf([aug.OpticalDistortion(), aug.GridDistortion(distort_limit=0.07)]),
                     'noise': aug.OneOf([aug.ISONoise(intensity=(0.2, 0.35)), aug.GaussNoise(var_limit=(20.0, 35.0)),
                                         aug.MultiplicativeNoise()]),
                     'blur': aug.OneOf([aug.GaussianBlur(blur_limit=(5, 7)), aug.GlassBlur(max_delta=9),
                                        aug.MedianBlur(blur_limit=9)]),
                     'crop': aug.RandomResizedCrop(height=target_shape[0], width=target_shape[1], scale=(0.55, 1.0)),
                     'rotate': aug.geometric.rotate.Rotate(limit=40)}
    for augmentation in config:
        augmentation_list.append(augmentations[augmentation])
    transform = aug.Compose(augmentation_list)
    return transform
