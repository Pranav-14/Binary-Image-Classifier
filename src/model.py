import torch
import torch.nn as nn
import torchvision.models as models

class TransferLearningClassifier(nn.Module):
    """
    High-Performance PyTorch Transfer Learning Model for Binary Image Classification
    Base Architecture: MobileNetV3-Small pre-trained on ImageNet.
    Input: (B, 3, 256, 256)
    Output: Sigmoid Class Probability [0.0 - 1.0] (0: Clean, 1: Garbage / Litter)
    """
    def __init__(self, pretrained=True):
        super(TransferLearningClassifier, self).__init__()
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            backbone = models.mobilenet_v3_small(weights=weights)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(1024, 1)
            )
            self.model = backbone
            self.architecture = "MobileNetV3-TransferLearning"
        except Exception as e:
            # Fallback to custom ResNet-like lightweight CNN
            self.model = self._build_custom_cnn()
            self.architecture = "Custom-Lightweight-CNN"

    def _build_custom_cnn(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

def build_model(pretrained=True):
    return TransferLearningClassifier(pretrained=pretrained)
