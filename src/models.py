import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class SimCLR(nn.Module):
    def __init__(self, base_model: str = 'resnet18', out_dim: int = 128, num_classes: int=None):
        super().__init__()
        
        if base_model == 'resnet18':
            self.backbone = resnet18(weights=None)
        elif base_model == 'resnet50':
            self.backbone = resnet50(weights=None)
        else:
            raise ValueError(f"Backbone {base_model} not supported")

        dim_mlp = self.backbone.fc.in_features

        # Remove FC layer
        self.backbone.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, out_dim) 
        )
        
        self.num_classes = num_classes 
        # Classifier head for the supervised accuracy 
        if num_classes is not None:
            self.classifier = nn.Linear(dim_mlp, num_classes) 
        else: 
            self.classifier = nn.Identity()

        def forward(self, x, return_features=False):
        h = self.backbone(x)
        
        
        if return_features:
            return h

        if not self.training and self.num_classes is not None:
            return self.classifier(h)

        # training with SupCon
        if self.num_classes is not None:
            z = self.projection(h)
            z = F.normalize(z, dim=1)
            class_logits = self.classifier(h)
            return z, class_logits
        
        # Default: Training SimCLR 
        z = self.projection(h)
        return F.normalize(z, dim=1)
