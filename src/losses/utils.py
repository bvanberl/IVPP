from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

def all_gather(
        tensor: Tensor,
        num_replicas: Optional[int] = None
):
    """Gathers a tensor from replicas on other devices

    :param tensor: Tensor to get replicas for
    :param num_replicas: Number of other devices
    """
    num_replicas = dist.get_world_size() if num_replicas is None else num_replicas
    other_replica_tensors = [torch.zeros_like(tensor) for _ in range(num_replicas)]
    dist.all_gather(other_replica_tensors, tensor)
    return torch.cat(other_replica_tensors, dim=0)

def multiclass_accuracy(logits: Tensor, labels: Tensor):
    """Calculates the accuracy of predicted logits.

    :param logits: Unnormalized predictions, shape (N, C)
    :param labels: labels, shape (N)
    :return: Accuracy value
    """

    # Apply softmax to the predictions if they are not already probabilities
    pred_probs = torch.softmax(logits, dim=1)

    # Get the class with the highest probability for each sample
    _, pred_classes = pred_probs.max(dim=1)

    # Calculate accuracy by comparing predicted classes to true labels
    correct = torch.sum(pred_classes == labels).item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

def off_diagonal(X: Tensor) -> Tensor:
    """
    Assembles the off-diagonal entries of square matrix X and flattens the matrix
    :param X: A square matrix, with shape (w, w)
    :return: A 1D tensor of `X`'s off-diagonal entries, with shape (w*(w - 1))
    """
    h, w = X.shape
    assert h == w
    return X.flatten()[:-1].view(h - 1, h + 1)[:, 1:].flatten()


def weighted_moments(
        X: Tensor,
        dim: int,
        weights: Tensor
) -> (Tensor, Tensor):
    """Compute weighted moments along axis

    Determines the weighted mean and variance along the
    desired axis. Implemented according to definition at
    https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weighvar.pdf
    :param X: Matrix, with shape (h, w)
    :param dim: Samples axis
    :param weights: Weights per sample, with shape (`X.shape[dim]`)
    :return: weighted mean, weighted variance
    """

    N = X.shape[dim]    # Number of data points
    assert N == weights.shape[0]
    N_prime = torch.count_nonzero(weights)  # Number of nonzero weights
    weights = torch.unsqueeze(weights, dim=1)

    # Calculate the weighted mean
    weight_sum = torch.sum(weights)
    weighted_mean = torch.sum(X * weights, dim=dim) / weight_sum

    # Calculate the squared differences from the mean
    squared_diff = torch.square(X - weighted_mean)

    # Calculate the weighted variance
    numerator = torch.sum(squared_diff * weights, dim=dim)
    denominator = (N_prime - 1.) * weight_sum / N_prime
    weighted_var = numerator / denominator

    return weighted_mean, weighted_var