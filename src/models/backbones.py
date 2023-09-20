import torch
from torch.nn import Module, Sequential, Identity
from torchvision.models import resnet18, efficientnet_v2_s, \
    mobilenet_v3_small, vgg16

# Paths of pretrained weights
RESNET18_WEIGHTS = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
EFFICIENTNETV2S_WEIGHTS = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"
MOBILENETV3S_WEIGHTS = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"
VGG16_WEIGHTS = "https://download.pytorch.org/models/vgg16-397923af.pth"


def get_backbone(
        backbone_name: str,
        imagenet_weights: bool = False,
        n_cutoff_layers: int = 0
) -> Module:
    '''Initializes the desired backbone.
    :param backbone_name: A string in the list below specifying the model
    :param imagenet_weights: Flag indicating whether to initialize layers
        with ImageNet-pretrained weights. If False, weights will be randomly
        initialized and bias units will be disabled.
    :param cutoff_layer: Number of layers to remove from the end of the
        backbone model.
    :return: TensorFlow model callable for the backbone, yet to be compiled
    '''

    backbone_name = backbone_name.lower()
    if backbone_name == 'resnet18':
        backbone = resnet18()
        backbone = load_backbone_weights(
            backbone,
            imagenet_weights,
            RESNET18_WEIGHTS,
            n_cutoff_layers
        )
    elif backbone_name == 'efficientnetv2b0':
        backbone = efficientnet_v2_s()
        backbone = load_backbone_weights(
            backbone,
            imagenet_weights,
            EFFICIENTNETV2S_WEIGHTS,
            n_cutoff_layers
        )
    elif backbone_name == 'mobilenetv3':
        backbone = mobilenet_v3_small()
        backbone = load_backbone_weights(
            backbone,
            imagenet_weights,
            MOBILENETV3S_WEIGHTS,
            n_cutoff_layers
        )
    elif backbone_name == 'vgg16':
        backbone = vgg16()
        backbone = load_backbone_weights(
            backbone,
            imagenet_weights,
            VGG16_WEIGHTS,
            n_cutoff_layers
        )
    else:
        raise Exception(f"Unsupported backbone architecture: {backbone_name}")
    return backbone
