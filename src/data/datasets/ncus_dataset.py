import os
from typing import Optional, Callable

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import cv2


class NCUSDataset(Dataset):
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
        self.video_dirs = video_records["clip_dir"].tolist()
        self.video_ids = video_records["id"].tolist()
        self.img_counts = video_records["n_frames"].tolist()
        self.fps = video_records["fps"].tolist()
        self.img_root_dir = img_root_dir
        if channels == 1:
            self.img_read_flag = cv2.IMREAD_GRAYSCALE
        else:
            self.img_read_flag = cv2.IMREAD_COLOR
        self.max_time_delta = max_time_delta
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.sample_weights = sample_weights
        self.img_ext = img_ext
        self.cardinality = len(self.video_ids)

    def __len__(self):
        return self.cardinality

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        n_frames = self.img_counts[idx]
        fps = self.fps[idx]

        # Sample two images that are separated by no more than a threshold
        max_dur_frame_delta = int(self.max_time_delta * fps)
        max_frame_delta = np.minimum(max_dur_frame_delta, n_frames)
        img_idx1 = np.random.randint(0, n_frames, dtype=int)
        idx2_min = np.maximum(0, img_idx1 - max_frame_delta)
        idx2_max = np.minimum(img_idx1 + max_frame_delta, n_frames - 1)
        img_idx2 = np.random.randint(idx2_min, idx2_max + 1, dtype=int)

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
        if self.sample_weights:
            frame_delta = np.abs(img_idx2 - img_idx1)
            sw = (max_dur_frame_delta - frame_delta + 1.) / (max_dur_frame_delta + 1.)
        else:
            sw = 1.
        return x1, x2, sw

    def _img_path_from_record(self, video_dir: str, img_idx: int) -> str:
        return os.path.join(
            self.img_root_dir,
            video_dir,
            f"{img_idx:05d}{self.img_ext}"
        )
