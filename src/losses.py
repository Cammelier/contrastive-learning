import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, supervised=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.supervised = supervised

    def forward(self, features, labels=None):
        # L'indentazione deve essere di 8 spazi (o 2 tab) rispetto all'inizio del file
        device = features.device
        
        # 1. Normalization 
        features = F.normalize(features, dim=1)
        full_batch_size = features.shape[0]

        # 2. Logits (Similarity Matrix)
        logits = torch.matmul(features, features.T) / self.temperature

        # Stabilità numerica: LogSumExp trick
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 3. Costruzione della Positive Mask
        if labels is None or not self.supervised:
            # Modalità SimCLR (Self-supervised)
            # Aspetta [x_i, x_j] concatenati, quindi batch_size è la metà
            batch_size = full_batch_size // 2
            mask = torch.eye(batch_size, device=device).repeat(2, 2)
        else: 
            # Modalità SupCon (Supervised)
            labels = labels.contiguous().view(-1, 1)
            # Se hai passato il batch concatenato [x_i, x_j], raddoppia i labels
            if labels.shape[0] != full_batch_size:
                labels = labels.repeat(2, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            
        # 4. Rimuovi l'auto-similarità (diagonale) dai calcoli
        logits_mask = torch.ones_like(mask) - torch.eye(full_batch_size, device=device)
        mask = mask * logits_mask 

        # 5. Calcolo Log-Probabilità stabile
        exp_logits = torch.exp(logits) * logits_mask
        # Somma riga per riga + epsilon per evitare log(0)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)

        # 6. Media sulle coppie positive
        pos_per_row = mask.sum(1)
        pos_per_row_clamped = torch.clamp(pos_per_row, min=1)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_row_clamped

        # Calcolo finale della loss (solo righe con almeno un positivo)
        valid_rows = pos_per_row > 0
        if valid_rows.any():
            loss = -mean_log_prob_pos[valid_rows].mean()
        else:
            loss = -mean_log_prob_pos.mean() 

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma 
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        return focal_loss.sum()