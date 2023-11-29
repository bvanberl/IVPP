import os
from typing import List
import logging

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import wandb
import albumentations as A
import cv2

from src.data.datasets.ncus_dataset import NCUSBmodeDataset, NCUSMmodeDataset
from src.data.datasets.image_datasets import *
from src.data.augmentation.pipelines import *


def get_augmentation_transforms_pretrain(
        pipeline: str,
        height: int,
        width: int,
        **augment_kwargs
) -> (A.Compose, A.Compose):
    """Get augmentation transformation pipelines

    :param pipeline: Name of pipeline.
                     One of {'byol', 'ncus', 'uscl', or 'none'}
    :param height: Image height
    :param width: Image width
    :param pipeline_kwargs: Pipeline keyword arguments
    :return: Augmentation pipelines for first and second images
    """
    pipeline = pipeline.lower()
    if pipeline == "byol":
        return (
            get_byol_augmentations(height, width),
            get_byol_augmentations(height, width)
        )
    elif pipeline == "ncus":
        return (
            get_ncus_augmentations(height, width, **augment_kwargs),
            get_ncus_augmentations(height, width, **augment_kwargs)
        )
    elif pipeline == "uscl":
        return (
            get_uscl_augmentations(height, width),
            get_uscl_augmentations(height, width)
        )
    else:
        if pipeline != "none":
            logging.warning(f"Unrecognized augmentation pipeline: {pipeline}.\n"
                            f"No augmentations will be applied.")
        return (
            get_validation_scaling(height, width),
            get_validation_scaling(height, width),
        )


def prepare_pretrain_dataset(
        img_root: str,
        pretrain_method: str,
        us_mode: str,
        file_df: pd.DataFrame,
        batch_size: int,
        width: int,
        height: int,
        augment_pipeline: str = "ncus",
        shuffle: bool = False,
        channels: int = 1,
        n_workers: int = 0,
        world_size: int = 1,
        **preprocess_kwargs
) -> DataLoader:
    '''
    Constructs a US dataset for a joint embedding self-supervised pretraining task.
    :param img_root: Root directory in which all images are stored. Will be prepended to path in frames table.
    :param pretrain_method: Pretraining method
    :param us_mode: US Mode. Either 'bmode' or 'mmode'
    :param file_df: A table of US video or image properties
    :param batch_size: Batch size for pretraining
    :param width: Desired width of US images
    :param height: Desired height of US images
    :param augment: If True, applies data augmentation transforms to the inputs
    :param shuffle: Flag indicating whether to shuffle the dataset
    :param channels: Number of channels
    :param max_time_delta: Maximum temporal separation of two frames
    :param n_workers: Number of workers for preloading batches
    :param world_size: Number of processes. If 1, then not using distributed mode
    :param preprocess_kwargs: Keyword arguments for preprocessing
    :return: A batched dataset ready for iterating over preprocessed batches
    '''

    # Construct the dataset
    pretrain_method = pretrain_method.lower()
    augment1, augment2 = get_augmentation_transforms_pretrain(
        augment_pipeline,
        height,
        width,
        **preprocess_kwargs["augmentation"]
    )
    if pretrain_method in ["ncus_barlow_twins", "ncus_vicreg"]:
        if us_mode == 'mmode':
            dataset = NCUSMmodeDataset(
                file_df,
                img_root,
                channels,
                preprocess_kwargs["max_x_delta"],
                transforms1=augment1,
                transforms2=augment2,
                sample_weights=preprocess_kwargs["sample_weights"]
            )
        else:
            dataset = NCUSBmodeDataset(
                file_df,
                img_root,
                channels,
                preprocess_kwargs["max_time_delta"],
                transforms1=augment1,
                transforms2=augment2,
                sample_weights=preprocess_kwargs["sample_weights"]
            )
    elif pretrain_method in ["simclr", "barlow_twins", "vicreg"]:

        dataset = ImagePretrainDataset(
            file_df,
            img_root,
            channels,
            transforms1=augment1,
            transforms2=augment2,
        )
    else:
        raise NotImplementedError(f"{pretrain_method} has not been implemented.")

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = None
    else:
        sampler = None
    device_batch_size = batch_size // world_size
    data_loader = DataLoader(
        dataset,
        batch_size=device_batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader


def get_video_dataset_from_frames(
        frames_df: pd.DataFrame,
        clips_df: pd.DataFrame,
        clip_columns: List[str],
        rep_by_length: bool = False
) -> pd.DataFrame:
    """
    Condenses frames records to clip records, including the path to the folders
    containing their constituent frames, along with the number of frames in each
    video.
    :param frames_df: A set of frame-wise records. Columns must include
        "filepath" and "id". Images for the same video are expected to be
        in their own folder, named by clip id.
    :param clips_df: A set of frame-wise records. Columns must include
        "id" and those in `clip_columns`.
    :param clip_columns: A list of columns from `clips_df` to include in the
        resultant video DataFrame
    :param rep_by_length: If True, ensures video records are oversampled
        proportional to their length (in frames)
    :return: Video records DataFrame
    """
    print(frames_df.head())
    print(clips_df.head())

    agg_dict = {
        "filepath": "count",
        "id": "first",
    }
    for c in frames_df.columns:
        if "_label" in c:
            agg_dict.update({
                c: "first"
            })

    def get_clip_dir(path):
        if "\\" in path:
            return "/".join(path.split("\\")[:-1])
        else:
            return "/".join(path.split("/")[:-1])

    frames_df["clip_dir"] = frames_df["filepath"].apply(lambda path: get_clip_dir(path))
    frames_df.head()
    new_video_df = frames_df.groupby("clip_dir").agg(agg_dict).reset_index()
    new_video_df.rename(columns={"filepath": "n_frames"}, inplace=True)
    new_video_df = new_video_df.merge(clips_df[["id"] + clip_columns], how="left", on="id")
    if rep_by_length:
        min_n_frames = new_video_df["n_frames"].min()
        copies = (new_video_df["n_frames"] / min_n_frames).astype(int).tolist()
        new_video_df = new_video_df.loc[new_video_df.index.repeat(copies)].reset_index(drop=True)

    return new_video_df


def load_data_for_pretrain(
        image_dir: str,
        splits_dir: str,
        pretrain_method: str,
        batch_size: int,
        augment_pipeline: str = "ncus",
        use_unlabelled: bool = True,
        channels: int = 1,
        width: int = 224,
        height: int = 224,
        us_mode: str = "bmode",
        world_size: int = 1,
        n_workers: int = 0,
        **preprocess_kwargs
) -> (DataLoader, pd.DataFrame):
    """
    Retrieve data, data splits, and returns an iterable preprocessed dataset for pretraining
    :param cfg: The config.yaml file dictionary
    :param batch_size: Batch size for datasets
    :param run: The wandb run object that is initialized
    :param data_artifact_name: Artifact name for raw data and files
    :param data_version: Artifact version for raw data
    :param splits_artifact_name: Artifact name for train/val/test splits
    :param splits_version: Artifact version for train/val/test splits
    :param redownload_data: Flag indicating whether the dataset artifact should be redownloaded
    :param augment_pipeline: Augmentation strategy identifier
    :param use_unlabelled: Flag indicating whether to use the unlabelled data in
    :param channels: Number of channels
    :param max_pixel_val: Maximum value for pixel intensity
    :param width: Desired width of images
    :param height: Desired height of images
    :param us_mode: US Mode. Either 'bmode' or 'mmode'.
    :param world_size: Number of processes. If 1, then not using distributed mode
    :param n_workers: Number of workers for preloading batches
    :param preprocess_kwargs: Keyword arguments for preprocessing
    :return: dataset for pretraining
    """
    
    if us_mode == 'bmode':
        image_fn_suffix = 'frames'
    elif us_mode == 'mmode':
        image_fn_suffix = 'mmodes'
    else:
        raise Exception(f"Invalid US mode provided: {us_mode}")

    # Load data for pretraining
    labelled_train_frames_path = os.path.join(splits_dir, f'train_set_{image_fn_suffix}.csv')
    labelled_train_clips_path = os.path.join(splits_dir, 'train_set_clips.csv')
    if os.path.exists(labelled_train_frames_path) and os.path.exists(labelled_train_clips_path):
        labelled_train_frames_df = pd.read_csv(labelled_train_frames_path)
        labelled_train_clips_df = pd.read_csv(labelled_train_clips_path)
    else:
        labelled_train_frames_df = pd.DataFrame()
        labelled_train_clips_df = pd.DataFrame()
    unlabelled_frames_path = os.path.join(splits_dir, f'unlabelled_{image_fn_suffix}.csv')
    unlabelled_clips_path = os.path.join(splits_dir, 'unlabelled_clips.csv')
    if os.path.exists(unlabelled_frames_path) and os.path.exists(unlabelled_clips_path):
        unlabelled_frames_df = pd.read_csv(unlabelled_frames_path)
        unlabelled_clips_df = pd.read_csv(unlabelled_clips_path)
    else:
        unlabelled_frames_df = pd.DataFrame()
        unlabelled_clips_df = pd.DataFrame()
    val_frames_path = os.path.join(splits_dir, f'val_set_{image_fn_suffix}.csv')
    val_clips_path = os.path.join(splits_dir, 'val_set_clips.csv')
    if os.path.exists(val_frames_path) and os.path.exists(val_clips_path):
        val_frames_df = pd.read_csv(val_frames_path)
        val_clips_df = pd.read_csv(val_clips_path)
    else:
        val_frames_df = pd.DataFrame()
        val_clips_df = pd.DataFrame()

    if use_unlabelled:
        train_frames_df = pd.concat([labelled_train_frames_df, unlabelled_frames_df])
        train_clips_df = pd.concat([labelled_train_clips_df, unlabelled_clips_df])
    else:
        train_frames_df = labelled_train_frames_df
        train_clips_df = labelled_train_clips_df

    # Condense frame records to clip records
    if pretrain_method in ["simclr", "vicreg", "barlow_twins"]:
        train_df = train_frames_df
        val_df = val_frames_df
    else:
        train_df = get_video_dataset_from_frames(train_frames_df, train_clips_df, ["fps"])
        if val_frames_df.shape[0] > 0:
            val_df = get_video_dataset_from_frames(val_frames_df, val_clips_df, ["fps"])
        else:
            val_df = None

    train_set = prepare_pretrain_dataset(
        image_dir,
        pretrain_method,
        us_mode,
        train_df,
        batch_size,
        width,
        height,
        augment_pipeline=augment_pipeline,
        shuffle=True,
        channels=channels,
        world_size=world_size,
        n_workers=n_workers,
        **preprocess_kwargs
    )
    if val_frames_df.shape[0] > 0:
        val_set = prepare_pretrain_dataset(
            image_dir,
            pretrain_method,
            us_mode,
            val_df,
            batch_size,
            width,
            height,
            augment_pipeline="none",
            shuffle=False,
            channels=channels,
            distributed=False,
            n_workers=n_workers,
            **preprocess_kwargs
        )
    else:
        val_set = None

    return train_set, train_df, val_set, val_df


def prepare_labelled_dataset(image_df: pd.DataFrame,
                             img_root: str,
                             us_mode: str,
                             height: int,
                             width: int,
                             batch_size: int,
                             label_col: str,
                             shuffle: bool = False,
                             augment_pipeline: Optional[str] = None,
                             channels: int = 1,
                             n_classes: int = 2,
                             n_workers: int = 0,
                             world_size: int = 1,
                             **augment_kwargs
                             ):
    '''
    Constructs a dataset for a supervised learning task.
    :param image_df: A table of image properties. Each row corresponds to an US image.
                     Must contain "filepath" and label_col columns
    :param img_root: Root directory containing images
    :param us_mode: US mode ("bmode" or "mmode")
    :param height: Image height
    :param width: Image width
    :param batch_size: Batch size for pretraining
    :param label_col: Column name containing label for downstream task
    :param shuffle: Flag indicating whether to shuffle the dataset
    :param augment_pipeline: Augmentation pipeline identifier
    :param channels: Number of channels
    :param n_classes: Number of classes
    :param n_workers: Number of workers for loading images
    :param world_size: Number of processes. If 1, then not using distributed mode
    :param augment_kwargs: Keyword arguments for augmentations
    :return: A batched dataset loader
    '''
    image_paths = image_df["filepath"].tolist()
    if label_col in image_df.columns:
        labels = image_df[label_col].to_numpy(dtype=np.int64)
    else:
        print(f"No label column named {label_col}. Setting labels to 0.")
        labels = np.zeros(image_df.shape[0], dtype=float)
    if augment_pipeline == "ncus":
        transforms = get_ncus_augmentations(height, width, **augment_kwargs)
    elif augment_pipeline == "supervised_ncus":
        transforms = get_supervised_bmode_augmentions(height, width, **augment_kwargs)
    elif augment_pipeline == "uscl":
        transforms = get_uscl_augmentations(height, width, **augment_kwargs)
    else:
        if augment_pipeline != "none":
            logging.warning(f"Unrecognized augmentation pipeline: {augment_pipeline}.\n"
                            f"No augmentations will be applied.")
        transforms = get_validation_scaling(height, width)
    dataset = ImageClassificationDataset(
        img_root,
        image_paths,
        labels,
        channels,
        n_classes,
        transforms=transforms
    )
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = None
    else:
        sampler = None
    device_batch_size = batch_size // world_size
    data_loader = DataLoader(
        dataset,
        batch_size=device_batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        pin_memory=True,
        sampler=sampler
    )
    return data_loader


def load_data_supervised(cfg: dict,
                         batch_size: int,
                         us_mode: str,
                         label_col: str,
                         data_artifact_name: str,
                         splits_artifact_name: str,
                         image_dir: str = None,
                         splits_dir: str = None,
                         run: wandb.sdk.wandb_run.Run = None,
                         data_version: str = 'latest',
                         splits_version: str = 'latest',
                         redownload_data: bool = True,
                         percent_train: float = 1.0,
                         height: int = 224,
                         width: int = 224,
                         channels: int = 1,
                         augment_pipeline: str = "supervised_bmode",
                         seed: int = 0,
                         n_workers: int = 0,
                         fold: Optional[int] = None,
                         k: Optional[int] = None,
                         **augment_kwargs
    ) -> (DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''
    Retrieve data, data splits, and returns an iterable preprocessed dataset for supervised learning experiments
    :param cfg: The config.yaml file dictionary
    :param us_mode: US mode ("bmode" or "mmode")
    :param image_dir: Images directory
    :param splits_dir: Directory containing CSVs for each data split
    :param batch_size: Batch size for datasets
    :param label_col: The column in the images DataFrame corresponding to the label for the downstream task
    :param run: The wandb run object that is initialized
    :param data_artifact_name: Artifact name for raw data and files
    :param data_version: Artifact version for raw data
    :param splits_artifact_name: Artifact name for train/val/test splits
    :param splits_version: Artifact version for train/val/test splits
    :param redownload_data: Flag indicating whether the dataset artifact should be redownloaded
    :param percent_train: Proportion of train set to use for training. Integer in [1, 100]
    :param height: Image height
    :param width: Image width
    :param channels: Number of channels
    :param oversample_minority: True if the minority class is upsampled to balance class distribution, False otherwise
    :param seed: Random state that ensures replicable training set shuffling prior to taking `percent_train` of it
    :param n_workers: Number of workers for data loading
    :param fold: Test set fold identifier. Only set if executing k-fold cross validation.
    :param k: Total number of folds if executing k-fold cross validation.
    :return: (training set, validation set, test set, training set table, validation set table, test set table, )
    '''

    if us_mode == 'bmode':
        image_fn_suffix = 'frames'
    elif us_mode == 'mmode':
        image_fn_suffix = 'mmodes'
    else:
        raise Exception(f"Invalid US mode provided: {us_mode}")

    # Retrieve versioned dataset artifact
    if run is not None and redownload_data:
        data = run.use_artifact(f"{data_artifact_name}:{data_version}")
        splits = run.use_artifact(f"{splits_artifact_name}:{splits_version}")
        data_dir = data.download(os.path.join(cfg['WANDB']['LOCAL_DATA_DIR'], data_artifact_name))
        splits_dir = splits.download(os.path.join(cfg['WANDB']['LOCAL_DATA_DIR'], 'splits'))
    else:
        data_dir = image_dir
        splits_dir = splits_dir

    if fold is None:
        train_frames_df = pd.read_csv(os.path.join(splits_dir, f'train_set_{image_fn_suffix}.csv'))
        train_frames_df = train_frames_df.sample(frac=1.0, random_state=seed)
        if percent_train < 1.0:
            n_train_examples = int(percent_train * train_frames_df.shape[0])
            train_frames_df = train_frames_df.iloc[:n_train_examples]

        val_frames_path = os.path.join(splits_dir, f'val_set_{image_fn_suffix}.csv')
        val_frames_df = pd.read_csv(val_frames_path) if os.path.exists(val_frames_path) else pd.DataFrame()
        test_frames_path = os.path.join(splits_dir, f'test_set_{image_fn_suffix}.csv')
        test_frames_df = pd.read_csv(test_frames_path) if os.path.exists(test_frames_path) else pd.DataFrame()

        # Remove examples that do not have a label relevant for the current task
        train_frames_df = train_frames_df.loc[train_frames_df[label_col] != -1]
        val_frames_df = val_frames_df.loc[val_frames_df[label_col] != -1]
        test_frames_df = test_frames_df.loc[test_frames_df[label_col] != -1]
    else:
        test_frames_df = pd.read_csv(os.path.join(splits_dir, f'fold{fold}_{image_fn_suffix}.csv'))
        val_frames_df = test_frames_df
        train_frames_df = pd.DataFrame(columns=test_frames_df.columns)
        for i in range(1, k + 1):
            if i != fold:
                fold_df = pd.read_csv(os.path.join(splits_dir, f'fold{i}_{image_fn_suffix}.csv'))
                train_frames_df = pd.concat([train_frames_df, fold_df], axis=0)

    # For the lung sliding task, take only the brightest image from the original video
    # if label_col == 'lung_sliding_label':
    #     train_frames_df = train_frames_df.loc[train_frames_df['brightness_rank'] == 0]
    #     val_frames_df = val_frames_df.loc[val_frames_df['brightness_rank'] == 0]
    #     test_frames_df = test_frames_df.loc[test_frames_df['brightness_rank'] == 0]

    n_classes = train_frames_df[label_col].nunique()
    train_set = prepare_labelled_dataset(
        train_frames_df,
        data_dir,
        us_mode,
        height,
        width,
        batch_size,
        label_col,
        augment_pipeline=augment_pipeline,
        shuffle=True,
        channels=channels,
        n_classes=n_classes,
        n_workers=n_workers,
        **augment_kwargs
    )
    val_set = prepare_labelled_dataset(
        val_frames_df,
        data_dir,
        us_mode,
        height,
        width,
        batch_size,
        label_col,
        augment_pipeline="none",
        shuffle=False,
        channels=channels,
        n_classes=n_classes,
        n_workers=n_workers
    ) if not val_frames_df.empty else None
    test_set = prepare_labelled_dataset(
        test_frames_df,
        data_dir,
        us_mode,
        height,
        width,
        batch_size,
        label_col,
        augment_pipeline="none",
        shuffle=False,
        channels=channels,
        n_classes=n_classes
    ) if not test_frames_df.empty else None

    return train_set, val_set, test_set, train_frames_df, val_frames_df, test_frames_df

