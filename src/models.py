import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 128, hidden_dim: int = 512, num_classes: int = None):
        super().__init__()
        
        # Backbone: Comune a SimCLR e SupCon
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # Projection Head: Indispensabile per entrambi
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) 
        )
        
        self.num_classes = num_classes 
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_dim, num_classes) 

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection(h)
        # Normalizzazione L2: Obbligatoria per SimCLR e SupCon
        z = F.normalize(z, dim=1)
        
        # Restituiamo sempre h e z per evitare il ValueError: too many values to unpack
        return h, z

    def get_logits(self, x):
        """Utilizzato solo nel fine-tuning o valutazione multiclasse"""
        with torch.no_grad():
            h = self.backbone(x)
        return self.classifier(h)
