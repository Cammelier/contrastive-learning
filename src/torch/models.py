import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SimCLR(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = resnet18(weights=None)

        dim_mlp = self.backbone.fc.in_features

        # We want to estract feature
        self.backbone.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim) 
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        return h, z