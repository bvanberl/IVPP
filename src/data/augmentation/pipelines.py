from typing import List

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_normalize_transform(
        mean_pixel_val: List[float] = None,
        std_pixel_val: List[float] = None
) -> A.Normalize:
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
    return A.Normalize(
        mean=mean_pixel_val,
        std=std_pixel_val
    )


def get_supervised_bmode_augmentions(
        brightness_delta: float = 0.4,
        contrast_low: float = 0.6,
        contrast_high: float = 1.4,
    ) -> A.Compose:
    """Defines augmentation pipeline for supervised learning experiments.
    :param brightness_delta: Maximum brightness increase/decrease, in [0, 1]
    :param contrast_low: Lower bound for contrast transformation
    :param contrast_high: Upper bound for contrast transformation
    :return: Callable augmentation pipeline
    """
    return A.Compose([
        A.ColorJitter(
            brightness=brightness_delta,
            contrast=(contrast_low, contrast_high),
            saturation=0.,
            hue=0.,
            p=1.
        ),
        A.HorizontalFlip(p=0.5),
        ToTensorV2()    # Rescale to [0, 1] & convert to channels-first
    ])

def get_byol_augmentations(
        height: int,
        width: int,
        t_prime: bool,
) -> A.Compose:
    """
    Applies random data transformations according to the data augmentations
    procedure outlined in BYOL (https://arxiv.org/pdf/2006.07733.pdf),
    Appendix B.
    :param height: Image height
    :param width: Image width
    :param t_prime: True if image is to be sampled from distribution, T'
    :return: Callable augmentation pipeline
    """
    if t_prime:
        final_transforms = A.Compose([
            A.GaussianBlur(
                blur_limit=(23, 23),
                sigma_limit=(0.2, 1.0),
                p=0.1
            ),
            A.Solarize(threshold=128, p=0.2)
        ])
    else:
        final_transforms = A.GaussianBlur(
                blur_limit=(23, 23),
                sigma_limit=(0.2, 1.0),
                p=1.0
        )
    return A.Compose([
        A.RandomResizedCrop(
            height,
            width,
            scale=(0.08, 1.),
            p=1.
        ),
        A.HorizontalFlip(p=0.5),
        A.OneOrOther(
            A.Compose([
                A.RandomBrightness(0.4, p=0.8),
                A.RandomContrast(0.4, p=0.8)
            ]),
            A.Compose([
                A.RandomContrast(0.4, p=0.8),
                A.RandomBrightness(0.4, p=0.8)
            ])
        ),
        final_transforms,
        ToTensorV2()
    ])

def get_bmode_baseline_augmentations(
        height: int,
        width: int,
        crop_prob: float = 0.8,
        min_crop_area: float = 0.7,
        max_crop_area: float = 1.0,
        brightness_prob: float = 0.7,
        max_brightness: float = 0.25,
        contrast_prob: float = 0.7,
        max_contrast: float = 0.25,
        blur_prob: float = 0.2,
        min_blur_sigma: float = 0.1,
        max_blur_sigma: float = 2.0,
        max_gauss_filter_width: int = 5
) -> A.Compose:
    """Applies random transformations to input B-mode image.

    Possible transforms include random crop & resize, contrast
    change, Gaussian blur, and horizontal flip.
    :param height: Image height
    :param width: Image width
    :param crop_prob: Probability of random crop
    :param min_crop_area: Minimum area of cropped region
    :param max_crop_area: Maximum area of cropped region
    :param brightness_prob: Probability of brightness change
    :param max_brightness: Maximum brightness difference
    :param contrast_prob: Probability of contrast change
    :param max_contrast: Maximum contrast change
    :param blur_prob: Probability of Gaussian blur
    :param min_blur_sigma: Minimum blur kernel standard deviation
    :param max_blur_sigma: Maximum blur kernel standard deviation
    :param max_gauss_filter_width: Maximum blur filter width
    :return: Callable augmentation pipeline
    """
    return A.Compose([
        A.RandomResizedCrop(
            height,
            width,
            scale=(min_crop_area, max_crop_area),
            p=crop_prob
        ),
        A.HorizontalFlip(p=0.5),
        A.OneOrOther(
            A.Compose([
                A.RandomBrightness(max_brightness, p=brightness_prob),
                A.RandomContrast(max_contrast, p=contrast_prob)
            ]),
            A.Compose([
                A.RandomContrast(max_contrast, p=contrast_prob),
                A.RandomBrightness(max_brightness, p=brightness_prob)
            ])
        ),
        A.GaussianBlur(
            blur_limit=max_gauss_filter_width,
            sigma_limit=(min_blur_sigma, max_blur_sigma),
            p=blur_prob
        ),
        ToTensorV2(),
    ])
