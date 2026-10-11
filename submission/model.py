import torch
from torchvision import models, transforms

NUM_CLASSES = 5
IMAGE_SIZE = 224
MODEL_NAME = "efficientnet_b0"


def build_model(name=MODEL_NAME, pretrained=False):
    weights = "DEFAULT" if pretrained else None
    if name == "densenet121":
        model = models.densenet121(weights=getattr(models.DenseNet121_Weights, weights) if weights else None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, NUM_CLASSES)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(
            weights=getattr(models.MobileNet_V3_Large_Weights, weights) if weights else None
        )
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=getattr(models.EfficientNet_B0_Weights, weights) if weights else None
        )
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(
            weights=getattr(models.ConvNeXt_Tiny_Weights, weights) if weights else None
        )
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Model tidak dikenal: {name}")
    return model


def get_model():
    return build_model()


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
