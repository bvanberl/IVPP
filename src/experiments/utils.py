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
    elif 'SLURM_LOCALID' in os.environ:
        # rank = int(os.environ['SLURM_PROCID'])
        # gpu = rank % torch.cuda.device_count()
        ngpus_per_node = torch.cuda.device_count()
        local_rank = int(os.environ.get("SLURM_LOCALID")) 
        rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank
        gpu = local_rank

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(gpu)
        os.environ['WORLD_SIZE'] = str(world_size)
        init_method = dist_url
    else:
        rank = 0
        gpu = 0
        init_method = None
        print('Not using distributed mode.')
    print(f'From rank {rank}, device ID {gpu} -- initializing process group.')

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
    return gpu, rank


def restore_extractor(
        extractor: Module,
        checkpoint: str,
        use_wandb: bool = False,
        wandb_run: Optional[wandb.wandb_sdk.wandb_run.Run] = None,
        freeze_prefix: Union[str, List[str]] = None
) -> (Module, str):
    """Restores a serialized feature extractor's weights

    Loads the weights of a serialized feature extractor. Also returns
    an identifier for the method used to pretraing it.
    :param extractor: Model extractor
    :param checkpoint: Location of model weights (path or artifact ID)
    :param use_wandb: If True, wandb experiment tracking is in use
    :param wandb_run: Current wandb run
    :param freeze_prefix: Prefixes for layers to be frozen
    :return: Restored feature extractor, pretraining method
    """
    if os.path.exists(checkpoint):
        checkpoint_path = checkpoint
    elif use_wandb:
        extractor_artifact = checkpoint
        checkpoint_path = wandb_run.use_artifact(extractor_artifact).download()
    else:
        raise ValueError(f"Ensure that either {checkpoint} is a valid .pth path"
                         f"or that it is a valid artefact ID and that wandb is enabled.")
    state_dict = torch.load(checkpoint_path)
    extractor_key = "extractor" if "extractor" in state_dict else "backbone"
    extractor.load_state_dict(state_dict[extractor_key])
    if freeze_prefix:
        for name, param in extractor.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze_prefix):
                param.requires_grad = False
    pretrain_method = state_dict["pretrain_method"]
    return extractor, pretrain_method

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
    if n_classes == 2:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0.)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0.)
        metrics["specificity"] = recall_score(1 - y_true, 1 - y_pred, zero_division=0.)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0.)
    else:
        y_true = np.argmax(y_true, -1)
        for i in range(n_classes):
            y_true_class = (y_true == i).astype(np.int64)
            y_pred_class = (y_pred == i).astype(np.int64)
            metrics[f"class{i}_precision"] = precision_score(y_true_class, y_pred_class, zero_division=0.)
            metrics[f"class{i}_recall"] = recall_score(y_true_class, y_pred_class, zero_division=0.)
            metrics[f"class{i}_specificity"] = recall_score(1 - y_true_class, 1 - y_pred_class, zero_division=0.)
            metrics[f"class{i}_f1"] = f1_score(y_true_class, y_pred_class, zero_division=0.)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    return metrics

def check_model_improvement(metric_name, cur_metric_value, prev_best_metric_value):
    """
    Checks to see if the model has improved on the metric of interest, given the
    best previously recorded value for that metric.
    :param metric_name: The metric of interest
    :param cur_metric_value: The current value of the metric of interest
    :param prev_best_metric_value: The best value of the metric of interest
        observed so far.
    :return: (True if new value is better, Best value for metric of interest)
    """
    if ("loss" in metric_name and cur_metric_value < prev_best_metric_value) or \
            ("loss" not in metric_name and cur_metric_value > prev_best_metric_value):
        return True, cur_metric_value
    else:
        return False, prev_best_metric_value
