import os
from typing import Optional, Callable, List

import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class ImagePretrainDataset(Dataset):
    def __init__(
            self,
            img_records: pd.DataFrame,
            img_root_dir: str,
            channels: int,
            transforms1: Optional[Callable],
            transforms2: Optional[Callable],
            img_ext: str = ".jpg"
    ):
        self.image_paths = img_records["filepath"]
        self.img_root_dir = img_root_dir
        if channels == 1:
            self.img_read_flag = cv2.IMREAD_GRAYSCALE
        else:
            self.img_read_flag = cv2.IMREAD_COLOR
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.img_ext = img_ext
        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx: int):

        # Load and copy image
        image_path = self.image_paths[idx]
        x1 = cv2.imread(image_path, self.img_read_flag)
        x2 = np.copy(x1)

        # Apply data augmentation transforms
        if self.transforms1:
            x1 = self.transforms1(x1)["image"]
        if self.transforms2:
            x2 = self.transforms2(x2)["image"]
        return x1, x2


class ImageClassificationDataset(Dataset):
    def __init__(
            self,
            img_root_dir: str,
            img_paths: List[str],
            labels: np.ndarray,
            channels: int,
            n_classes: int,
            transforms: Optional[Callable],
            img_ext: str = ".jpg"
    ):
        assert len(img_paths) == len(labels), "Number of images and labels must match."
        self.img_root_dir = img_root_dir
        self.image_paths = img_paths
        if n_classes > 2:
            self.labels = one_hot(torch.from_numpy(labels), num_classes=n_classes)
        else:
            self.labels = np.expand_dims(labels, axis=-1).astype(np.float32)
        if channels == 1:
            self.img_read_flag = cv2.IMREAD_GRAYSCALE
        else:
            self.img_read_flag = cv2.IMREAD_COLOR
        self.transforms = transforms
        self.img_ext = img_ext
        self.cardinality = len(self.image_paths)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx: int):

        # Load image
        image_path = os.path.join(
            self.img_root_dir,
            self.image_paths[idx]
        )
        x = cv2.imread(image_path, self.img_read_flag)

        # Apply data augmentation transforms
        if self.transforms:
            x = self.transforms(x)

        y = self.labels[idx]
        return x, y
