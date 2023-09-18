from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist

from utils import all_gather, multiclass_accuracy, off_diagonal, \
    weighted_moments

class SimCLRLoss(nn.Module):

    def __init__(self, reduction: str = "mean", tau: float = 0.05):
        """
        :param reduction: Type of reduction
        :param tau: SimCLR temperature
        """
        super().__init__()
        assert reduction in ["mean", "sum"], "Reduction must be 'mean' or 'sum'"
        self.reduction = reduction
        self.tau = tau
        self.contrastive_accuracy = None
        self.LARGE_CONST = 1e9

    def forward(
            self,
            z1: Tensor,
            z2: Tensor,
            sw: Optional[Tensor] = None
    ):
        """
        Computes SimCLR loss.
        :param z1: Embedding 1, with shape [batch_size, z_dim]
        :param z2: Embedding 2, with shape [batch_size, z_dim]
        :param sw: Per-example loss scaling values, with shape [batch_size]
        :return: SimCLR loss per example, of shape [batch_size]
        """

        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        batch_size = z1.shape[0]
        emb_dim = z1.shape[1]

        num_replicas = dist.get_world_size()
        if num_replicas > 1 and self.training:
            z1_all = all_gather(z1, num_replicas=num_replicas)
            z2_all = all_gather(z2, num_replicas=num_replicas)
            z1_all = z1_all.reshape(-1, emb_dim)
            z2_all = z2_all.reshape(-1, emb_dim)

            # Create pseudo-labels
            replica_id = dist.get_rank()
            labels = torch.arange(batch_size, device=z1.device) + replica_id * batch_size
            labels = labels.type(torch.int64)
            batch_size_all = z1_all.shape[0]
            masks = F.one_hot(labels, batch_size_all).to(z1_all.device)
            labels = F.one_hot(labels, batch_size_all * 2).to(z1_all.device)
        else:
            z1_all = z1
            z2_all = z2
            masks = F.one_hot(torch.arange(batch_size), batch_size).to(z1.device)
            labels = F.one_hot(torch.arange(batch_size), batch_size * 2).to(z1.device)

        # Compute pairwise similarity and divide by temperature
        # Set diagonal to large negative num to ensure that
        # term for (z_i, z_i) is close to 0 in cross entropy.
        logits_11 = torch.matmul(z1, z1_all.T) / self.tau
        logits_11 = logits_11 - masks * self.LARGE_NUM
        logits_22 = torch.matmul(z2, z2_all.T) / self.tau
        logits_22 = logits_22 - masks * self.LARGE_NUM
        logits_12 = torch.matmul(z1, z2_all.T) / self.tau
        logits_21 = torch.matmul(z2, z1_all.T) / self.tau

        # Calculate per-example loss
        logits1 = torch.cat([logits_12, logits_11], 1)
        labels = torch.argmax(labels, -1)
        loss_a = F.cross_entropy(input=logits1, target=labels, reduction="none")
        logits2 = torch.cat([logits_21, logits_22], 1)
        loss_b = F.cross_entropy(input=logits2, target=labels, reduction="none")

        # Apply sample weights
        if sw:
            loss_a = torch.mul(loss_a, sw)
            loss_b = torch.mul(loss_b, sw)

        # Calculate total per-example loss
        loss = loss_a + loss_b

        # Apply loss reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        # Calculate contrastive accuracy
        self.contrastive_accuracy = multiclass_accuracy(
            torch.cat([logits1, logits2], dim=0),
            torch.cat([labels, labels], dim=0)
        )

        return loss

    def get_instance_vars(self):
        return {
            "contrastive_accuracy": self.contrastive_accuracy
        }


class BarlowTwinsLoss(nn.Module):

    def __init__(self, batch_size: int, lambda_: float = 0.005):
        """
        :param batch_size: Batch size across all replicas
        :param lambda_: Weight for redundancy reduction term
        """
        super().__init__()
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.inv_term = None    # Invariance term
        self.rr_term = None     # Redundancy reduction term
        self.MARGIN = 1e-12
        self.COL_STD_EPSILON = 1e-5

    def _invariance_term(self, ccm: Tensor) -> Tensor:
        """
        Computes the invariance term in Barlow Twins, which penalizes small diagonal entries in the cross-correlation
        matrix of the embeddings.
        :param ccm: The cross-correlation matrix, with shape (z_dim, z_dim)
        :return: Invariance term, with shape (1)
        """
        on_diag = 1.0 - torch.diagonal(ccm)
        on_diag = torch.square(on_diag)
        inv_term = torch.sum(on_diag)
        return inv_term

    def _redundancy_reduction_term(self, ccm: Tensor) -> Tensor:
        """
        Computes the redundancy reduction term in Barlow Twins,
        which penalizes large off-diagonal entries in the
        cross-correlation matrix of the embeddings.
        :param ccm: Cross-correlation matrix, with shape (z_dim, z_dim)
        :return: Redundancy reduction term, with shape (1)
        """
        off_diag = off_diagonal(ccm)
        off_diag = torch.square(off_diag)
        rr_term = torch.sum(off_diag)
        return rr_term

    def _standardize_columns(
            self,
            x: Tensor,
            sw: Optional[Tensor] = None
    ) -> Tensor:
        """Standardizes the columns of x.

        Subtracts the mean and divide by standard deviation, for each
        column. If sample weights are provided, uses the weighted mean
        and standard deviation.
        :param x: A matrix, with shape (batch_size, z_dim)
        :param sw: Per-example loss scaling values, with shape (batch_size)
        :return: A standardized x, with shape (batch_size, z_dim)
        """

        if sw:
            col_mean, col_var = weighted_moments(x, 0, sw)
            col_std = torch.sqrt(col_var)
        else:
            col_std, col_mean = torch.std_mean(x, dim=0)

        norm_col = (x - col_mean) / (col_std + self.COL_STD_EPSILON)
        return norm_col


    def call(
        self,
        z1: Tensor,
        z2: Tensor,
        sw: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        Computes the Barlow Twins loss
        :param z1: Embedding 1, with shape (batch_size, z_dim)
        :param z2: Embedding 2, with shape (batch_size, z_dim)
        :param sw: Per-example loss weights, with shape (batch_size)
        :return: Barlow Twins loss, of shape (1)
        """

        z1 = self.standardize_columns(z1, sw=sw)
        z2 = self.standardize_columns(z2, sw=sw)

        # Construct the cross-correlation matrix
        ccm = z1.T @ z2
        ccm = ccm / self.batch_size
        torch.distributed.all_reduce(ccm) # Sum across devices

        # Calculate invariance term
        inv_term = self.invariance_term(ccm)

        # Calculate redundancy reduction term
        rr_term = self.redundancy_reduction_term(ccm)

        # Calculate final loss as weighted sum of terms
        loss = inv_term + self.lambda_ * rr_term + self.margin

        # Track the components of the loss
        self.inv_term = inv_term
        self.rr_term = rr_term
        self._metrics["redundancy_reduction_term"].update_state(rr_term)

        return loss

    def get_instance_vars(self):
        return {
            "invariance_term": self.inv_term,
            "redundancy_reduction_term": self.rr_term
        }


class VICRegLoss(nn.Module):

    def __init__(
            self,
            batch_size: int,
            lambda_: float = 25.,
            mu: float = 25.,
            nu: float = 1.,
            epsilon: float = 1e-4,
            gamma: float = 1.
    ):
        """
        :param batch_size: Batch size across all replicas
        :param lambda_: Weight for redundancy reduction term
        """
        super().__init__()
        self.batch_size = batch_size
        self.lambda_ = lambda_
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.gamma = gamma
        self.inv_term = None    # Invariance term
        self.var_term = None    # Variance term
        self.cov_term = None    # Covariance term

    def _invariance_term(
            self,
            z1: Tensor,
            z2: Tensor,
            sw: Optional[Tensor] = None
    ) -> Tensor:
        """Computes the invariance term in VICReg
        Computes the mean squared error between each pair of embeddings,
        averaged over the batch. Optionally applies sample weights.
        :param z1: First embeddings, with shape (batch_size, z_dim)
        :param z2: Second embeddings, with shape (batch_size, z_dim)
        :param sw: Per-example weighting values, with shape (batch_size)
        :return: Similarity loss, with shape (1)
        """
        if sw:
            sq_diffs = torch.unsqueeze(sw, dim=-1) * torch.square(z2 - z1)
            return torch.mean(sq_diffs)
        else:
            return F.mse_loss(z1, z2)

    def _variance_term(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Computes the VICReg variance term

        Computes the variance term in VICReg, which penalizes
        small standard deviation along each individual
        dimension of the embeddings. Expects embeddings to be
        mean-centered with respect to columns.
        :param z1: First embeddings, with shape (batch_size, z_dim)
        :param z2: Second embeddings, with shape (batch_size, z_dim)
        :return: Variance loss, with shape (1)
        """
        std_z1 = torch.sqrt(z1.var(dim=0) + self.epsilon)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.epsilon)
        reg_std_z1 = torch.mean(F.relu(1 - std_z1))
        reg_std_z2 = torch.mean(F.relu(1 - std_z2))
        return 0.5 * (reg_std_z1 + reg_std_z2)

    def _covariance_term(
            self,
            z1: Tensor,
            z2: Tensor,
            sw: Optional[Tensor] = None
    ) -> Tensor:
        """Computes the covariance term in VICReg

        Calculates the sum of the squared off-diagonal entries of
        the covariance matrix of each embedding. Computed for each
        set of embeddings and summed to produce the result.
        :param z1: First embeddings, with shape (batch_size, z_dim)
        :param z2: Second embeddings, with shape (batch_size, z_dim)
        :return: Covariance loss, with shape (1)
        """
        cov_z1 = (z1.T @ z1) / (self.batch_size - 1.)
        cov_z2 = (z2.T @ z2) / (self.batch_size - 1.)

        off_diag_z1_sum = torch.sum(torch.square(off_diagonal(z1)))
        off_diag_z2_sum = torch.sum(torch.square(off_diagonal(z2)))
        z_dim = z1.shape[-1]
        return (off_diag_z1_sum + off_diag_z2_sum) / z_dim

    def call(
        self,
        z1: Tensor,
        z2: Tensor,
        sw: Optional[Tensor] = None
    ) -> Tensor:
        """
        Computes the VICReg loss
        :param z1: Embedding 1, with shape (batch_size, z_dim)
        :param z2: Embedding 2, with shape (batch_size, z_dim)
        :param sw: Per-example loss weights, with shape (batch_size)
        :return: VICReg loss, of shape (1)
        """

        # Calculate invariance term
        self.inv_term = self._invariance_term(z1, z2, sw=sw)

        # Assemble embeddings across all devices
        z1 = all_gather(z1)
        z2 = all_gather(z2)

        # Subtract column mean from each element
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # Calculate variance term
        self.var_term = self._variance_term(z1, z2)

        # Calculate covariance term
        self.cov_term = self._covariance_term(z1, z2)

        # Calculate final loss as weighted sum of terms
        loss = self.lambda_ * self.inv_term + \
               self.mu * self.var_term + \
               self.nu * self.cov_term

        return loss

    def get_instance_vars(self):
        return {
            "invariance_term": self.inv_term,
            "variance_term": self.var_term,
            "covariance_term": self.cov_term
        }


