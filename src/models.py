import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50

class SimCLR(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 128, hidden_dim: int = 512, num_classes: int = None):
        super().__init__()
        
        # 1. Backbone: MLP 
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        

        # Capture feature dimension before removing the original FC layer
        dim_mlp = hidden_dim
        

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
            
        z = self.projection(h) 

        if not self.training and self.num_classes is not None:
            return self.classifier(h)

        return h, z

