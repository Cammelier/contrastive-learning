import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class SimCLR(nn.Module):
    def __init__(self, base_model: str = 'resnet18', out_dim: int = 128, num_classes: int = None):
        super().__init__()
        
        # 1. Initialize Backbone
        if base_model == 'resnet18':
            self.backbone = resnet18(weights=None)
            # Modify for STL-10 (96x96): kernel 3x3 and no maxpool to keep spatial resolution
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()
        elif base_model == 'resnet50':
            self.backbone = resnet50(weights=None)
            # Modify for STL-10: first conv 3x3 to preserve features from 96x96 images
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Note: ResNet50 usually keeps maxpool, but can be replaced if resolution drops too fast
        else:
            raise ValueError(f"Backbone {base_model} not supported")

        # Capture feature dimension before removing the original FC layer
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # 2. Projection Head (MLP) 
        # Crucial for SupCon/SimCLR: maps features to a latent space for the contrastive loss
        self.projection = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, out_dim), 
            nn.BatchNorm1d(out_dim) 
        )
        
        self.num_classes = num_classes 
        # 3. Classifier Head
        # Used for online linear probing during training and final inference
        if num_classes is not None:
            self.classifier = nn.Linear(dim_mlp, num_classes) 
        else: 
            self.classifier = nn.Identity()

    def forward(self, x, return_features=False):
        # Extract features from backbone (h)
        h = self.backbone(x)
        
        # Global Average Pooling (ensure shape is [B, dim_mlp])
        if len(h.shape) > 2:
            h = F.adaptive_avg_pool2d(h, (1, 1))
            h = torch.flatten(h, 1)

        if return_features:
            return h
            
        if not self.training and self.num_classes is not None:
            return self.classifier(h)

        z = self.projection(h)
        
        
        return h, z

