from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
from torch.utils.data.dataloader import DataLoader

def visualize_views(
        views: npt.NDArray,
        labels: npt.NDArray = None,
        predictions: npt.NDArray = None,
        num_imgs: int = 16,
        views_per_col: int = 8,
        fig_size: Tuple[int, int] = (24, 4),
        max_pixel_value: float = 1.0,
        min_pixel_value: float = 0.0,
        sample_weight: bool = False
):
    '''Visualizes multiple positive pairs on same plot

    Display side by side different image views with labels, and predictions
    :param views: Array of views
    :param views: Predictions from a model
    :param labels: image labels
    :param num_imgs: Number of images to view
    :param views_per_col: Number of images in one row. Defaults to 3.
    :param fig_size: Dimensions of the resulting figure (width, height)
    :param max_pixel_value: Max expected value for a pixel.
    :param min_pixel_value: Min expected value for a pixel.
    :param sample_weight: Display sample weight for each positive pair
    '''

    n_channels = views[0].shape[-1]
    colour_map = 'gray' if n_channels == 1 else 'viridis'
    num_views = len(views)
    num_imgs = num_imgs if num_imgs else len(views[0])
    num_col = views_per_col
    num_row = num_imgs // num_col
    num_row = num_row + 1 if num_imgs % num_col else num_row

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=fig_size)
    for i in range(num_imgs):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        scale = abs(max_pixel_value - min_pixel_value)
        pair = [(views[j][i] - min_pixel_value) / scale for j in range(num_views)]
        divider = np.ones((views[0][i].shape[0], 3, views[0][i].shape[-1])) * max_pixel_value
        ax.imshow(np.concat([pair[0], divider, pair[1]], axis=1), cmap=colour_map)
        if len(views) > 2 and sample_weight:
            ax.set_title(f"sw: {views[2]}")
        ax.set_axis_off()

        label = labels[i] if labels else i

        if predictions:
            ax.set_title("Label: {} | Pred: {:.5f}".format(label, predictions[i][0]))
        elif labels:
            ax.set_title("Label: {}".format(label))

        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()


def visualize_joint_views_dataset(
        ds: DataLoader,
        num_imgs: int = 16,
        views_per_col: int = 8,
        fig_size: Tuple[int, int] = (24, 4),
        max_pixel_value: float = 255.0,
        min_pixel_value: float = 0.0
):
    '''Produces views for a dataset of positive pairs

    :param ds: dataset that returns two perturbed views of an image
    :param num_imgs: Number of images to view
    :param views_per_col: Number of images in one row
    :param fig_size: Dimensions of the resulting figure (width, height)
    :param max_pixel_value: Max expected value for a pixel. Used to scale the image between [0,1].
    :param min_pixel_value: Min expected value for a pixel. Used to scale the image between [0,1].
    '''

    imgs = next(iter(ds))
    visualize_views(
        views=imgs,
        num_imgs=num_imgs,
        views_per_col=views_per_col,
        fig_size=fig_size,
        max_pixel_value=max_pixel_value,
        min_pixel_value=min_pixel_value
    )