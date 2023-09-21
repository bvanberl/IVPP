from typing import List, Tuple, Optional, Dict
from abc import abstractmethod

import torch
from torch.nn import Module
from torch import Tensor

from src.models.backbones import get_backbone
from src.models.projectors import get_projector

class JointEmbeddingModel(Module):

    def __init__(
            self,
            input_shape: Tuple[int, int, int, int],
            backbone_name: str,
            imagenet_weights: bool,
            projector_nodes: List[int],
            loss: Module,
            backbone_cutoff_layers: int = 0,
            projector_bias: bool = False
        ):
        """
        :param input_shape: Expected shape of input images (B, C, H, W)
        :param backbone_name: Backbone identifier
        :param imagenet_weights: If True, initializes backbone
            with ImageNet-pretrained weights
        :param projector_nodes: Number of nodes in each fully connected layer
        :param backbone_cutoff_layers: Number of layers to remove from the end of the
            backbone model.
        :param projector_bias: If True, use biases in fully connected layers
        """
        super().__init__()
        self.backbone = get_backbone(
            backbone_name,
            imagenet_weights,
            backbone_cutoff_layers
        )

        self.h_dim = self.backbone(torch.randn(*input_shape)).shape[-1]
        self.projector = get_projector(
            self.h_dim,
            projector_nodes,
            use_bias=projector_bias
        )
        self.loss= loss

    def forward(
            self, 
            x0: Tensor, 
            x1: Tensor,
            sw: Optional[Tensor] = None
    ) -> Tensor:

        # Compute features
        h0 = self.backbone(x0)
        h1 = self.backbone(x1)

        # Compute embeddings
        z0 = self.projector(h0)
        z1 = self.projector(h1)

        # Calculate loss
        loss = self.loss(z0, z1, sw)

        return loss
