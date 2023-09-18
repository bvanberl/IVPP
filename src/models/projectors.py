from typing import List

from torch.nn import Linear, BatchNorm1d, ReLU, Sequential

def get_projector(
            input_dim: int,
            fc_nodes: List[int],
            use_bias: bool = False,
    ) -> Sequential:
    """Initializes a projector MLP.

    Produces a multi-layer perceptron that acts as the projector
    :param input_dim: Dimensionality of input vector
    :param fc_nodes: Number of nodes in each fully connected layer
    :param use_bias: If True, use biases in fully connected layers
    :return:
    """
    assert len(fc_nodes) >= 2, "Projector must have >= 2 layers"
    layers = []
    layers.append(Linear(input_dim, fc_nodes[0], bias=use_bias))
    for i in range(1, len(fc_nodes) - 1):
        layers.append(Linear(fc_nodes[i - 1], fc_nodes[i], bias=use_bias))
        layers.append(BatchNorm1d(fc_nodes[i]))
        layers.append(ReLU(True))
    layers.append(Linear(fc_nodes[-2], fc_nodes[-1], bias=False))
    return Sequential(*layers)
