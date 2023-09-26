from typing import Dict, Optional, Tuple, List, Union, Callable
import os

import torch
from torch.nn import Module, Linear, ReLU, Sigmoid, Softmax, Sequential
from torch.utils.data.dataloader import DataLoader
import wandb
import numpy as np
import numpy.typing as npt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from torch.utils.tensorboard import SummaryWriter

def log_scalars(
        split: str,
        metrics_dict: Dict[str, float],
        step: int = None,
        writer: Optional[SummaryWriter] = None,
        use_wandb: bool = False
):

    for m in metrics_dict:
        if step is not None:
            writer.add_scalar(f"{m}/{split}", metrics_dict[m], step)
        else:
            writer.add_text(f"{m}/{split}", str(metrics_dict[m]))
        if use_wandb:
            if m not in ["epoch", "lr"]:
                wandb.log({f"{split}/{m}": metrics_dict[m]}, step=step)
            else:
                wandb.log({f"{m}": metrics_dict[m]}, step=step)

def init_distributed_mode(
        backend: str = "gloo",
        world_size: int = 1,
        dist_url: str = "env://"
):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        init_method = dist_url
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['WORLD_SIZE'] = str(world_size)
        init_method = dist_url
    else:
        rank = 0
        gpu = 0
        init_method = None
        print('Not using distributed mode.')

    torch.cuda.set_device(gpu)
    print('Distributed init (rank {}): {}, gpu {}'.format(
        rank, dist_url, gpu), flush=True)
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    torch.distributed.barrier()


def restore_backbone(
        backbone: Module,
        checkpoint: str,
        use_wandb: bool = False,
        wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None
) -> (Module, str):
    """Restores a serialized feature extractor's weights

    Loads the weights of a serialized feature extractor. Also returns
    an identifier for the method used to pretraing it.
    :param backbone: Model backbone
    :param checkpoint: Location of model weights (path or artifact ID)
    :param use_wandb: If True, wandb experiment tracking is in use
    :param wandb_run: Current wandb run
    :return: Restored feature extractor, pretraining method
    """
    if os.path.exists(checkpoint):
        checkpoint_path = checkpoint
    elif use_wandb:
        backbone_artifact = checkpoint
        checkpoint_path = wandb_run.use_artifact(backbone_artifact).download()
    else:
        raise ValueError(f"Ensure that either {checkpoint} is a valid .pth path"
                         f"or that it is a valid artefact ID and that wandb is enabled.")
    state_dict = torch.load(checkpoint_path)
    backbone.load_state_dict(state_dict["backbone"])
    pretrain_method = state_dict["pretrain_method"]
    return backbone, pretrain_method

def normal_init_linear(
        in_dim: int,
        out_dim: int,
        w_mean: float = 0.,
        w_std: float = 0.
):
    """
    Initializes a fully connected layer with weights randomly
    drawn from a Gaussian distribution with specified
    weight mean and std dev. The layer is initialized with zero
    bias as well.
    :param in_dim: Dimension of input
    :param out_dim: Dimension of output
    :param w_mean: Mean for weight matrix
    :param w_std: Std for weight matrix
    :return: Linear layer
    """
    layer = Linear(in_dim, out_dim)
    layer.weight.data.normal_(mean=w_mean, std=w_std)
    layer.bias.data.zero_()
    return layer

def get_classifier_head(
            input_dim: int,
            fc_nodes: List[int],
            n_classes: int
    ) -> Sequential:
    """Initializes a classification head.

    Produces a MLP classification head, complete with
    and fully connected layers.
    :param input_dim: Dimension of input vector
    :param fc_nodes: Number of nodes in each hidden layer.
    :param n_classes: Number of classes
    :return:
    """

    output_dim = 1 if n_classes <= 2 else n_classes
    layers = []

    # Add hidden and output fully connected layers
    if len(fc_nodes) > 0:
        layers.append(normal_init_linear(input_dim, fc_nodes[0]))
        layers.append(ReLU())
        for i in range(1, len(fc_nodes) - 1):
            layers.append(normal_init_linear(fc_nodes[i], fc_nodes[i + 1]))
            layers.append(ReLU())
        layers.append(normal_init_linear(fc_nodes[-1], output_dim))
    else:
        layers.append(normal_init_linear(input_dim, output_dim))

    # If two classes, output is a sigmoid unit; otherwise, softmax.
    return Sequential(*layers)

def get_classification_metrics(
        n_classes: int,
        y_prob: npt.NDArray,
        y_pred: npt.NDArray,
        y_true: npt.NDArray
) -> Dict[str, Union[float, int]]:
    """Returns classification metrics for predictions

    Gets classification metrics such as accuracy,
    precision, recall, specificity, AUC, and
    confusion matrix counts.
    :param n_classes: Number of classes
    :param y_prob: Predicted probabilities
    :param y_pred: Predicted classes
    :param y_true: Class labels
    :return: Dictionary of class metrics
    """
    metrics = {}
    metrics["auc"] = roc_auc_score(y_true, y_prob)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    if n_classes == 2:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0.)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0.)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0.)
    else:
        for i in range(n_classes):
            y_true_class = (y_true == i).astype(np.int64)
            y_pred_class = (y_pred == i).astype(np.int64)
            metrics["class{i}_precision"] = precision_score(y_true_class, y_pred_class, zero_division=0.)
            metrics["class{i}_recall"] = recall_score(y_true_class, y_pred_class, zero_division=0.)
            metrics["class{i}_f1"] = f1_score(y_true_class, y_pred_class, zero_division=0.)
    return metrics

