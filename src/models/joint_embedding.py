from typing import List, Tuple, Optional, Dict
from abc import abstractmethod

import torch
from torch.nn import Module
from torch import Tensor

from src.models.extractors import get_extractor
from src.models.projectors import get_projector

class JointEmbeddingModel(Module):

    def __init__(
            self,
            input_shape: Tuple[int, int, int, int],
            extractor_name: str,
            imagenet_weights: bool,
            projector_nodes: List[int],
            extractor_cutoff_layers: int = 0,
            projector_bias: bool = False
        ):
        """
        :param input_shape: Expected shape of input images (B, C, H, W)
        :param extractor_name: Extractor identifier
        :param imagenet_weights: If True, initializes extractor
            with ImageNet-pretrained weights
        :param projector_nodes: Number of nodes in each fully connected layer
        :param extractor_cutoff_layers: Number of layers to remove from the end of the
            extractor model.
        :param projector_bias: If True, use biases in fully connected layers
        """
        super().__init__()
        self.extractor = get_extractor(
            extractor_name,
            imagenet_weights,
            extractor_cutoff_layers
        )

        self.h_dim = self.extractor(torch.randn(*input_shape)).shape[-1]
        self.projector = get_projector(
            self.h_dim,
            projector_nodes,
            use_bias=projector_bias
        )

    def forward(
            self, 
            x0: Tensor, 
            x1: Tensor
    ) -> Tensor:

        # Compute features
        h0 = self.extractor(x0)
        h1 = self.extractor(x1)

        # Compute embeddings
        z0 = self.projector(h0)
        z1 = self.projector(h1)

        return z0, z1
