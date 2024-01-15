from typing import List

import torch
import torchvision
from torchvision.transforms import ToTensor, v2

torchvision.disable_beta_transforms_warning()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_normalize_transform(
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
) -> v2.Normalize:
    """Creates a pixel normalization transformation,

    Produces a pixel normalization transformation that
    scales pixel values to a desired mean and standard
    deviation. Defaults to ImageNet values.
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Normalization transform
    """
    if mean_pixel_val is None:
        mean_pixel_val = IMAGENET_MEAN
    if std_pixel_val is None:
        std_pixel_val = IMAGENET_STD
    return v2.Normalize(mean=mean_pixel_val, std=std_pixel_val)


def get_validation_scaling(
        height: int,
        width: int,
        resize_first: bool = False,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
) -> v2.Compose:
    """Defines augmentation pipeline for supervised learning experiments.
    :param height: Image height
    :param width: Image width
    :param resize_first: If True, resize image to (height, width) before transforms
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.ToImage(),
        v2.Resize((height, width), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if resize_first:
        transforms.insert(1, v2.Resize((height, width), antialias=True))
    return v2.Compose(transforms)


def get_supervised_bmode_augmentions(
        height: int,
        width: int,
        brightness_delta: float = 0.4,
        contrast_low: float = 0.6,
        contrast_high: float = 1.4,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None,
    ) -> v2.Compose:
    """Defines augmentation pipeline for supervised learning experiments.
    :param height: Image height
    :param width: Image width
    :param brightness_delta: Maximum brightness increase/decrease, in [0, 1]
    :param contrast_low: Lower bound for contrast transformation
    :param contrast_high: Upper bound for contrast transformation
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((height, width), antialias=True),
        v2.ColorJitter(
            brightness=brightness_delta,
            contrast=(contrast_low, contrast_high),
            saturation=0.,
            hue=0.
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        get_normalize_transform(mean_pixel_val, std_pixel_val)
    ])

def get_uscl_augmentations(
    height: int,
    width: int,
    resize_first: bool = False,
    min_crop_area: float = 0.8,
    max_crop_area: float = 1.0,
    min_crop_ratio: float = 0.8,
    max_crop_ratio: float = 1.25,
    mean_pixel_val: List[float] = None,
    std_pixel_val: List[float] = None,
):
    """Defines augmentation pipeline for supervised learning experiments.

    Same pipeline as used in USCL: https://arxiv.org/pdf/2011.13066.pdf
    :param height: Image height
    :param width: Image width
    :param resize_first: If True, resize image to (height, width) before transforms
    :param min_crop_area: Minimum area of cropped region
    :param max_crop_area: Maximum area of cropped region
    :param min_crop_ratio: Minimum aspect ratio (w:h) for cropped region
    :param max_crop_ratio: Maximum aspect ratio (w:h) for cropped region
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.ToImage(),
        v2.RandomResizedCrop(
            (height, width),
            scale=(min_crop_area, max_crop_area),
            ratio=(min_crop_ratio, max_crop_ratio),
            antialias=True
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if resize_first:
        transforms.insert(1, v2.Resize((height, width), antialias=True))
    return v2.Compose(transforms)

def get_byol_augmentations(
        height: int,
        width: int,
        resize_first: bool = False,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None,
) -> v2.Compose:
    """
    Applies random data transformations according to the data augmentations
    procedure outlined in VICReg (https://arxiv.org/pdf/2105.04906.pdf),
    Appendix C.1, which is derived from BYOL.
    :param height: Image height
    :param width: Image width
    :param resize_first: If True, resize image to (height, width) before transforms
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.ToImage(),
        v2.RandomResizedCrop((height, width), scale=(0.08, 1.), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0., 0.)], p=0.8),
        v2.RandomApply([v2.GaussianBlur(23)], p=0.8),
        v2.RandomSolarize(0.5, p=0.1),
        v2.ToDtype(torch.float32, scale=True),
        get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if resize_first:
        transforms.insert(1, v2.Resize((height, width), antialias=True))
    return v2.Compose(transforms)



def get_ncus_augmentations(
        height: int,
        width: int,
        resize_first: bool = False,
        min_crop_area: float = 0.4,
        max_crop_area: float = 1.0,
        min_crop_ratio: float = 3. / 4.,
        max_crop_ratio: float = 4. / 3.,
        brightness_prob: float = 0.5,
        max_brightness: float = 0.25,
        contrast_prob: float = 0.7,
        max_contrast: float = 0.25,
        blur_prob: float = 0.2,
        min_blur_sigma: float = 0.1,
        max_blur_sigma: float = 2.0,
        gauss_filter_width: int = 5,
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
):
    """Applies random transformations to input B-mode image.

    Possible transforms include random crop & resize, contrast
    change, Gaussian blur, and horizontal flip.
    :param height: Image height
    :param width: Image width
    :param resize_first: If True, resize image to (height, width) before transforms
    :param min_crop_area: Minimum area of cropped region
    :param max_crop_area: Maximum area of cropped region
    :param min_crop_ratio: Minimum aspect ratio (w:h) for cropped region
    :param max_crop_ratio: Maximum aspect ratio (w:h) for cropped region
    :param brightness_prob: Probability of brightness change
    :param max_brightness: Maximum brightness difference
    :param contrast_prob: Probability of contrast change
    :param max_contrast: Maximum contrast change
    :param blur_prob: Probability of Gaussian blur
    :param min_blur_sigma: Minimum blur kernel standard deviation
    :param max_blur_sigma: Maximum blur kernel standard deviation
    :param gauss_filter_width: Maximum blur filter width
    :param mean_pixel_val: Channel-wise means
    :param std_pixel_val: Channel-wise standard deviation
    :return: Callable augmentation pipeline
    """
    transforms = [
        v2.ToImage(),
        v2.RandomResizedCrop(
            (height, width),
            scale=(min_crop_area, max_crop_area),
            ratio=(min_crop_ratio, max_crop_ratio),
            antialias=True
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(max_brightness, 0., 0., 0.)], p=brightness_prob),
        v2.RandomApply([v2.ColorJitter(0., max_contrast, 0., 0.)], p=contrast_prob),
        v2.RandomApply([
            v2.GaussianBlur(gauss_filter_width, (min_blur_sigma, max_blur_sigma))],
            p=blur_prob),
        v2.ToDtype(torch.float32, scale=True),
       get_normalize_transform(mean_pixel_val, std_pixel_val)
    ]
    if resize_first:
        transforms.insert(1, v2.Resize((height, width), antialias=True))
    return v2.Compose(transforms)
