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


def get_extractor(
        extractor_name: str,
        imagenet_weights: bool = False,
        n_cutoff_layers: int = 0
) -> Module:
    '''Initializes the desired extractor.
    :param extractor_name: A string in the list below specifying the model
    :param imagenet_weights: Flag indicating whether to initialize layers
        with ImageNet-pretrained weights. If False, weights will be randomly
        initialized and bias units will be disabled.
    :param n_cutoff_layers: Number of layers to remove from the end of the
        extractor model.
    :return: TensorFlow model callable for the extractor, yet to be compiled
    '''

    extractor_name = extractor_name.lower()
    if extractor_name == 'resnet18':
        model = get_resnet18(imagenet_weights, n_cutoff_layers)
    elif extractor_name == 'resnet14':
        model = get_resnet14(imagenet_weights, n_cutoff_layers)
    elif extractor_name == 'efficientnetv2b0':
        model = get_efficientnetv2s(imagenet_weights, n_cutoff_layers)
    elif extractor_name == 'mobilenetv3':
        model = get_mobilenetv3s(imagenet_weights, n_cutoff_layers)
    elif extractor_name == 'vgg16':
        model = get_vgg16(imagenet_weights, n_cutoff_layers)
    else:
        raise Exception(f"Unsupported extractor architecture: {extractor_name}")
    return model

def get_resnet18(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    extractor = resnet18()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(RESNET18_WEIGHTS)
        extractor.load_state_dict(state_dict)
    extractor.fc = Identity()
    if n_cutoff_layers > 0:
        extractor = Sequential(*list(extractor.children())[:-(n_cutoff_layers + 1)])
    return extractor

def get_resnet14(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    extractor = resnet18()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(RESNET18_WEIGHTS)
        extractor.load_state_dict(state_dict)
    extractor.layer4 = Identity()
    extractor.fc = Identity()
    if n_cutoff_layers > 0:
        extractor = Sequential(*list(extractor.children())[:-(n_cutoff_layers + 1)])
    return extractor

def get_efficientnetv2s(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    extractor = efficientnet_v2_s()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(EFFICIENTNETV2S_WEIGHTS)
        extractor.load_state_dict(state_dict)
    extractor.fc = Identity()
    if n_cutoff_layers > 0:
        extractor = Sequential(*list(extractor.children())[:-(n_cutoff_layers + 1)])
    return extractor

def get_mobilenetv3s(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    extractor = mobilenet_v3_small()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(MOBILENETV3S_WEIGHTS)
        extractor.load_state_dict(state_dict)
    extractor.classifier = Identity()
    extractor.fc = Identity()
    if n_cutoff_layers > 0:
        extractor = Sequential(*list(extractor.children())[:-(n_cutoff_layers + 1)])
    return extractor

def get_vgg16(
    imagenet_weights: bool = False,
    n_cutoff_layers: int = 0
) -> Module:
    extractor = vgg16()
    if imagenet_weights:
        state_dict = torch.hub.load_state_dict_from_url(VGG16_WEIGHTS)
        extractor.load_state_dict(state_dict)
    extractor.avgpool = AdaptiveAvgPool2d(output_size=(1,1))
    extractor.classifier = Identity()
    if n_cutoff_layers > 0:
        extractor = Sequential(*list(extractor.children())[:-(n_cutoff_layers + 1)])
    return extractor