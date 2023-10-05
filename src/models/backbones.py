import torch
from torch.nn import Module, Sequential, Identity, AdaptiveAvgPool2d
import torchvision
from torchvision.models import resnet18, efficientnet_v2_s, \
    mobilenet_v3_small, vgg16

torchvision.disable_beta_transforms_warning()

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
    :param n_cutoff_layers: Number of layers to remove from the end of the
        backbone model.
    :return: TensorFlow model callable for the backbone, yet to be compiled
    '''

    backbone_name = backbone_name.lower()
    if backbone_name == 'resnet18':
        model = get_resnet18(imagenet_weights, n_cutoff_layers)
    elif backbone_name == 'resnet14':
        model = get_resnet14(imagenet_weights, n_cutoff_layers)
    elif backbone_name == 'efficientnetv2b0':
        model = get_efficientnetv2s(imagenet_weights, n_cutoff_layers)
    elif backbone_name == 'mobilenetv3':
        model = get_mobilenetv3s(imagenet_weights, n_cutoff_layers)
    elif backbone_name == 'vgg16':
        model = get_vgg16(imagenet_weights, n_cutoff_layers)
    else:
        raise Exception(f"Unsupported backbone architecture: {backbone_name}")
    return model

def get_resnet18(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    backbone = resnet18()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(RESNET18_WEIGHTS)
        backbone.load_state_dict(state_dict)
    backbone.fc = Identity()
    if n_cutoff_layers > 0:
        backbone = Sequential(*list(backbone.children())[:-(n_cutoff_layers + 1)])
    return backbone

def get_resnet14(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    backbone = resnet18()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(RESNET18_WEIGHTS)
        backbone.load_state_dict(state_dict)
    backbone.layer4 = Identity()
    backbone.fc = Identity()
    if n_cutoff_layers > 0:
        backbone = Sequential(*list(backbone.children())[:-(n_cutoff_layers + 1)])
    return backbone

def get_efficientnetv2s(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    backbone = efficientnet_v2_s()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(EFFICIENTNETV2S_WEIGHTS)
        backbone.load_state_dict(state_dict)
    backbone.fc = Identity()
    if n_cutoff_layers > 0:
        backbone = Sequential(*list(backbone.children())[:-(n_cutoff_layers + 1)])
    return backbone

def get_mobilenetv3s(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    backbone = mobilenet_v3_small()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(MOBILENETV3S_WEIGHTS)
        backbone.load_state_dict(state_dict)
    backbone.classifier = Identity()
    backbone.fc = Identity()
    if n_cutoff_layers > 0:
        backbone = Sequential(*list(backbone.children())[:-(n_cutoff_layers + 1)])
    return backbone

def get_vgg16(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    backbone = vgg16()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(VGG16_WEIGHTS)
        backbone.load_state_dict(state_dict)
    backbone.avgpool = AdaptiveAvgPool2d(output_size=(1,1))
    backbone.classifier = Identity()
    if n_cutoff_layers > 0:
        backbone = Sequential(*list(backbone.children())[:-(n_cutoff_layers + 1)])
    return backbone