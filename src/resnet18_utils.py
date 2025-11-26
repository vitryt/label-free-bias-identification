import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
# Original ResNet paper: https://arxiv.org/abs/1512.03385

class WaterbirdsResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True): # Pre-trained weights on ImageNet-1K
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)