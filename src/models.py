import torch 
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, input_dim_num, cat_dims, out_dim=128, hidden_dim=512, num_classes=None):
        super().__init__()
        
        # 1. Embedding Layers per ogni colonna categorica
        # cat_dims è una lista di tuple: [(n_cat_1, emb_dim_1), (n_cat_2, emb_dim_2), ...]
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n, embedding_dim=d) 
            for n, d in cat_dims
        ])
        
        # Calcola dimensione totale input dopo embedding
        total_emb_dim = sum([d for _, d in cat_dims])
        total_input_dim = input_dim_num + total_emb_dim
        
        # 2. Backbone
        self.backbone = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # 3. Projection Head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim) 
        )
        
        self.num_classes = num_classes 
        if num_classes is not None:
            self.classifier = nn.Linear(hidden_dim, num_classes) 

    def _process_input(self, x):
        """Helper interno per gestire embeddings e concatenazione"""
        x_num, x_cat = x
        
        # Processa categorici
        embedded_cats = []
        for i, emb_layer in enumerate(self.embeddings):
            # x_cat[:, i] è la colonna i-esima del batch
            embedded_cats.append(emb_layer(x_cat[:, i]))
            
        # Concatena tutto: [Batch, Num_Features + Emb_Features]
        x_cat_flat = torch.cat(embedded_cats, dim=1)
        
        # Unisci numerici e categorici processati
        x_full = torch.cat([x_num, x_cat_flat], dim=1)
        return x_full

    def forward(self, x):
        # Usa l'helper per preparare l'input del backbone
        x_full = self._process_input(x)
        
        h = self.backbone(x_full)
        z = self.projection(h)
        z = F.normalize(z, dim=1)
        
        return h, z

    def get_logits(self, x):
        """
        Utilizzato nel fine-tuning o valutazione.
        Deve processare l'input (tupla) esattamente come il forward.
        """
        # Non serve torch.no_grad() qui se lo chiami durante il training (es. fine-tuning),
        # ma se è solo per inferenza puoi metterlo esternamente.
        
        x_full = self._process_input(x)
        h = self.backbone(x_full)
        
        return self.classifier(h)