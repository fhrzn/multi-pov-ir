from torch import nn
import torch
from torchvision import models


class ClassifierHead(nn.Module):
    def __init__(self, n_feature: int, n_classes: int, dropout: float = 0.2):
        super().__init__()

        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feature, n_classes))

    def forward(self, x: torch.Tensor):
        return self.cls(x)


class ViewpointClassifier(nn.Module):
    """This viewpoint classifier following the implementation of
    Maksimova et al. (2024) on Viability of Zero-shot Classification
    and Search of Historical Photos paper.
    """

    def __init__(self, nclass: int, weights: str = "IMAGENET1K_V1"):
        super().__init__()

        self.base_model = models.resnet50(weights=weights)

        # freeze params
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.classifier = ClassifierHead(self.base_model.fc.in_features, nclass)
        self.base_model.fc = self.classifier

    def forward(self, x: torch.Tensor):
        return self.base_model(x)

    