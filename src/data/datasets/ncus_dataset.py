import os
from typing import Optional, Callable
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2


class NCUSDataset(Dataset, metaclass=ABCMeta):
    """
    A dataset that produces positive pairs of images that 
    originate from the same US video.
    :attr video_dirs: Directories containing images from each video
    :attr video_ids: IDs for each video
    :attr frame_counts: Number of images in each video
    :attr img_root_dir: The root directory for all images
    :attr img_read_flag: Image reading flag indicating format
    :attr transforms1: Set of stochastic transforms to apply to the first
          image in each pair
    :attr transforms2: Set of stochastic transforms to apply to the second
          image in each pair
    :attr sample_weights: If True, sample weights are returns for each pair
    :attr img_ext: File extension for images
    :attr cardinality: Number of videos in the dataset
    """
    
    def __init__(
            self,
            video_records: pd.DataFrame,
            img_root_dir: str,
            channels: int,
            transforms1: Optional[Callable],
            transforms2: Optional[Callable],
            sample_weights: bool = True,
            img_ext: str = ".jpg"
    ):
        """
        :param video_records: DataFrame containing at least the following columns:
               'clip_dir', 'id'
        :param img_root_dir: The root directory for all images
        :param channels: Number of channels in each image
        :param transforms1: Set of stochastic transforms to apply to the first
               image in each pair
        :param transforms2: Set of stochastic transforms to apply to the second
               image in each pair
        :param sample_weights: If True, sample weights are returns for each pair
        :param img_ext: File extension for images
        """
        self.video_dirs = video_records["clip_dir"].tolist()
        self.video_ids = video_records["id"].tolist()
        self.frame_counts = video_records["n_frames"].tolist()
        self.img_root_dir = img_root_dir
        if channels == 1:
            self.img_read_flag = cv2.IMREAD_GRAYSCALE
        else:
            self.img_read_flag = cv2.IMREAD_COLOR
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.sample_weights = sample_weights
        self.img_ext = img_ext
        self.cardinality = len(self.video_ids)

    def __len__(self):
        """
        Returns the number of videos in the dataset.

        :return: Cardinality of the dataset
        """
        return self.cardinality

    def __getitem__(self, idx):
        """
        Samples a positive pair of images from the dataset.

        Loads two random images from the same video subject to
        separation constraints and applies stochastic data
        augmentation to each, producing a positive pair of images.
        Calculates and returns a sample weight for the positive pair.
        :param idx: Index of the video in the dataset
        :return: (image 1, image 2, sample weight)
        """
        video_dir = self.video_dirs[idx]
        img_idx1, img_idx2, sw = self._get_image_idxs_and_sw(idx)

        # Load images
        x1_path = self._img_path_from_record(video_dir, img_idx1)
        x2_path = self._img_path_from_record(video_dir, img_idx2)
        x1 = cv2.imread(x1_path, self.img_read_flag)
        x2 = cv2.imread(x2_path, self.img_read_flag)

        # Apply data augmentation transforms
        if self.transforms1:
            x1 = self.transforms1(x1)
        if self.transforms2:
            x2 = self.transforms2(x2)

        # Determine sample weight according to distance between images
        if not self.sample_weights:
            sw = 1.
        return x1, x2, sw

    def _img_path_from_record(self, video_dir: str, img_idx: int) -> str:
        """
        Returns the path to an image

        :param video_dir: Directory containing a video's images
        :param img_idx: The image index within the video
        :return: The path to the `img_idx`th image in the video
        """
        return os.path.join(
            self.img_root_dir,
            video_dir,
            f"{img_idx:05d}{self.img_ext}"
        )

    @abstractmethod
    def _get_image_idxs_and_sw(self, clip_idx) -> (int, int, float):
        """
        Determines indices and sample weight for images within the video

        :param clip_idx: Index of clip in the dataset of clips
        :return: (image index of first image, image index of second image,
                  sample weight)
        """
        return -1, -1, -1.


class NCUSBmodeDataset(NCUSDataset):
    """
    NCUS dataset for B-modes. Pairs of images are sampled such that
    their temporal separation in the clip is no greater than a
    threshold.

    :attr fps: Frames per second in each video
    :attr max_time_delta: Maximum seconds between images in a positive pair
    """

    def __init__(
            self,
            video_records: pd.DataFrame,
            img_root_dir: str,
            channels: int,
            max_time_delta: float,
            transforms1: Optional[Callable],
            transforms2: Optional[Callable],
            sample_weights: bool = True,
            img_ext: str = ".jpg"
    ):
        super().__init__(
            video_records,
            img_root_dir,
            channels,
            transforms1=transforms1,
            transforms2=transforms2,
            sample_weights=sample_weights,
            img_ext=img_ext
        )
        self.fps = video_records["fps"].tolist()
        self.max_time_delta = max_time_delta

    def _get_image_idxs_and_sw(self, clip_idx) -> (int, int, float):
        """
        Determines indices and sample weight for images within the video

        Samples two images within the video at `clip_idx` that are
        temporally separated by no more than `self.max_time_delta` seconds.
        Calculates the sample weight, which is inversely proportional
        to the temporal separation between the sampled clips.
        :param clip_idx: Index of clip in the dataset of clips
        :return: (image index of first image, image index of second image,
                  sample weight)
        """
        n_frames = self.frame_counts[clip_idx]
        fps = self.fps[clip_idx]

        # Sample two images that are separated by no more than a threshold
        max_dur_frame_delta = int(self.max_time_delta * fps)
        max_frame_delta = np.minimum(max_dur_frame_delta, n_frames)
        img_idx1 = np.random.randint(0, n_frames, dtype=int)
        idx2_min = np.maximum(0, img_idx1 - max_frame_delta)
        idx2_max = np.minimum(img_idx1 + max_frame_delta, n_frames - 1)
        img_idx2 = np.random.randint(idx2_min, idx2_max + 1, dtype=int)

        frame_delta = np.abs(img_idx2 - img_idx1)
        sw = (max_dur_frame_delta - frame_delta + 1.) / (max_dur_frame_delta + 1.)

        return img_idx1, img_idx2, sw


class NCUSMmodeDataset(NCUSDataset):
    """
    NCUS dataset for M-modes. Pairs of images are sampled such that
    their horizontal distance in the clip is no greater than a
    threshold. Assumes images were initially resized to a standard
    dimension.

    :attr max_x_delta: Maximum horizontal distance, in pixels, between
          positive pairs of M-modes
    """
    def __init__(
            self,
            video_records: pd.DataFrame,
            img_root_dir: str,
            channels: int,
            max_x_delta: int,
            transforms1: Optional[Callable],
            transforms2: Optional[Callable],
            sample_weights: bool = True,
            img_ext: str = ".jpg"
    ):
        """
        :param video_records: DataFrame containing at least the following columns:
               'clip_dir', 'id', 'n_frames', 'fps', 'brightness_rank'.
        :param img_root_dir: The root directory for all images
        :param channels: Number of channels in each image
        :param max_x_delta: Maximum horizontal distance, in pixels, between
               positive pairs of M-modes
        :param transforms1: Set of stochastic transforms to apply to the first
               image in each pair
        :param transforms2: Set of stochastic transforms to apply to the second
               image in each pair
        :param sample_weights: If True, sample weights are returns for each pair
        :param img_ext: File extension for images
        """
        super().__init__(
            video_records,
            img_root_dir,
            channels,
            transforms1=transforms1,
            transforms2=transforms2,
            sample_weights=sample_weights,
            img_ext=img_ext
        )
        self.max_x_delta = max_x_delta
        self.img_idxs = video_records['img_idx'].tolist()
        self.mmode_counts = video_records['mmode_count'].tolist()

    def _get_image_idxs_and_sw(self, clip_idx) -> (int, int, float):
        """
        Determines indices and sample weight for images within the video

        Samples two images within the video at `clip_idx` that are
        horizontally separated by no more than `self.max_x_delta` pixels.
        Calculates the sample weight, which is inversely proportional
        to the horizontal distance between the sampled clips.
        :param clip_idx: Index of clip in the dataset of clips
        :return: (image index of first image, image index of second image,
                  sample weight)
        """
        n_frames = self.mmode_counts[clip_idx]
        idx_list = self.img_idxs[clip_idx]
        
        # Sample two images that are separated by no more than a threshold
        img_idx1 = np.random.choice(idx_list)
        idx2_min = np.maximum(0, img_idx1 - self.max_x_delta)
        idx2_max = np.minimum(img_idx1 + self.max_x_delta, n_frames - 1)
        idx2_choices = [i for i in list(range(idx2_min, idx2_max + 1)) if i in idx_list]
        img_idx2 = np.random.choice(idx2_choices)

        # Calculate sample weight
        x_delta = np.abs(img_idx2 - img_idx1)
        sw = (self.max_x_delta - x_delta + 1.) / (self.max_x_delta + 1.)

        return img_idx1, img_idx2, sw