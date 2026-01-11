import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

class WaterbirdsResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True): # Also loading weights trained on ImageNet-1K
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)