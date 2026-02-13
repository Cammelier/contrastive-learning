import torch 
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 128, hidden_dim: int = 512, num_classes: int = None):
        super().__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Projection Head:
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) 
        )
        
        self.num_classes = num_classes 
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_dim, num_classes) 

    def forward(self, x, mode='contrastive'):
        # 1. Feature extraction
        h = self.backbone(x)
        
        
        if mode == 'contrastive':
            z = self.projection(h)
            return F.normalize(z, dim=1)
        
        if mode == 'classify':
            return self.classifier(h)
            
        return h 
