import argparse
import os
from typing import Sequence, List

import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
import torchvision
torchvision.disable_beta_transforms_warning()

from src.data.utils import prepare_labelled_dataset
from src.models.extractors import get_backbone
from src.experiments.utils import restore_backbone

def get_features(
        backbone_path: str,
        image_df_path: str,
        image_dir: str,
        label_col: str,
        feats_path: str,
        channels: int = 3,
        backbone_type: str = "mobilenetv3",
        n_cutoff_layers: int = 0,
        batch_size: int = 256,
        height: int = 224,
        width: int = 224
):
    """Obtains predicted features for a feature extractor

    Loads desired backbone (pretrained or not) and produces
    features for the labelled dataset saved at image_df_path.
    Saves the features as a numpy array. If features already
    exist at feats_path, loads and returns those instead.
    :param backbone_path: Path to pretrained model. If
        'imagenet', loads supervised pretrained weights.
    :param image_df_path: Dataset spreadsheet path
    :param image_dir: Path to directory containing images
    :param label_col: Column of spreadsheet containing label
    :param feats_path: Path at which to save features as .npy
    :param channels: Number of image channels
    :param backbone_type: Backbone architecture to be used
        when not loading a pretrained model
    :param n_cutoff_layers: Number of layers to remove from the end of the
        backbone model.
    :param batch_size: Batch size for prediction
    :param height: Image height
    :param width: Image width
    """

    if os.path.exists(feats_path):
        print(f"Features already saved at {feats_path}")
        feats = np.load(feats_path)
    else:
        # Load feature extractor
        if backbone_path == "imagenet":
            backbone = get_backbone(backbone_type, True, n_cutoff_layers)
        else:
            backbone = get_backbone(backbone_type, False, n_cutoff_layers)
            backbone, _ = restore_backbone(backbone, backbone_path)
        backbone = backbone.cuda()
        backbone.eval()

        # Produce labelled dataset
        image_df = pd.read_csv(image_df_path)
        n_classes = image_df[label_col].nunique()
        ds = prepare_labelled_dataset(
            image_df,
            image_dir,
            height,
            width,
            batch_size,
            label_col,
            channels=channels,
            n_classes=n_classes
        )

        # Predict features for all images in the dataset
        feats = []
        with torch.no_grad():
            for _, (x,_) in enumerate(tqdm(ds)):
                x = x.cuda()
                h = backbone.forward(x)
                feats.append(h.cpu())
        feats = np.concatenate(feats, axis=0)

        # Save the representations
        feats_dir = os.path.dirname(feats_path)
        if not os.path.exists(feats_dir):
            os.makedirs(feats_dir)
        np.save(feats_path, feats)
    return feats

def plot_embedding(
        emb: np.ndarray,
        labels: Sequence[int],
        label_names: List[str],
        save_path: str
):
    """Scales and plots 2D embeddings, coloured by class
    :param emb: 2D embeddings for all examples in dataset, shape (N, 2)
    :param labels: List of N labels for each example
    :param label_names: List of label names; one for each class
    :param save_path: Path at which to save resulting plot
    """
    _, ax = plt.subplots()
    emb = MinMaxScaler().fit_transform(emb)

    for l in range(len(label_names)):
        ax.scatter(
            *emb[labels == l].T,
            marker=f"o",
            s=2,
            color=plt.cm.Dark2(l),
            alpha=1.0,
            zorder=2,
            label=label_names[l]
        )
    ax.axis("off")
    ax.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.05), markerscale=10, ncol=2)
    plt.savefig(save_path)

def do_tsne(
        label_col: str,
        test_df_path: str,
        feats: np.ndarray,
        save_dir: str,
        label_names: List[str]
):
    """Produces and plots t-SNE embeddings for features.

    Calculates 2D t-SNE embeddings for feature outputted by a
    pretrained model and plots them.
    :param label_col: Column of spreadsheet containing label
    :param image_df_path: Dataset spreadsheet path
    :param feats: Features for all examples in dataset, shape (N, H)
    :param save_dir: Path at which to save the plots
    :param label_names: List of label names; one for each class
    """

    print(f"Features for all examples in the dataset have shape {feats.shape}")
    test_df = pd.read_csv(test_df_path)
    labels = test_df[label_col].to_numpy()
    labelled_indices = np.where(labels != -1)
    labels = labels[labelled_indices]
    feats = feats[labelled_indices]

    print(f"Executing t-SNE for {feats.shape[0]} examples with label {label_col}")
    tsne = TSNE(n_components=2, n_iter=500)
    Z = tsne.fit_transform(feats)
    save_path = os.path.join(save_dir, f"tsne_{label_col}.png")
    plot_embedding(Z, labels, label_names, save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--backbone', required=True, default='imagenet', type=str, help='Path to pretrained model')
    parser.add_argument('--backbone_type', required=True, default='resnet18', type=str, help='Architecture of backbone')
    parser.add_argument('--df_path', required=True, type=str, help='Path to image dataset spreadsheet')
    parser.add_argument('--image_dir', required=True, type=str, help='Path to image data root directory')
    parser.add_argument('--label_col', required=True, type=str, help='Label column')
    parser.add_argument('--feats_path', required=True, type=str, help='Path at which features are saved or to be saved')
    parser.add_argument('--label_names', required=False, type=str, nargs='+', default=['negative', 'positive'], help='Class identifiers')
    args = vars(parser.parse_args())

    backbone_dir = args["backbone"]
    backbone_type = args["backbone_type"]
    df_path = args["df_path"]
    image_dir = args["image_dir"]
    label_col = args["label_col"]
    feats_path = args["feats_path"]
    label_names = args["label_names"]

    feats = get_features(backbone_dir, df_path, image_dir, label_col, feats_path, backbone_type=backbone_type)
    plot_save_dir = os.path.dirname(feats_path)
    do_tsne(label_col, df_path, feats, plot_save_dir, label_names)
