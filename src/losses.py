import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, supervised=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.supervised = supervised

    def forward(self, features, labels=None):
        device = features.device
        
        # 1. Normalization 
        features = F.normalize(features, dim=1)
        full_batch_size = features.shape[0]

        # 2. Logits
        logits = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 3. Positive mask
        if labels is None or not self.supervised:
            # SimCLR (Self-supervised)
            batch_size = full_batch_size // 2
            mask = torch.eye(batch_size, device=device).repeat(2, 2)
        else: 
            #  SupCon (Supervised Contrastive)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != full_batch_size:
                labels = labels.repeat(2, 1)
            
            # binary mask
            mask = torch.eq(labels, labels.T).float().to(device)
            
       
        logits_mask = torch.ones_like(mask) - torch.eye(full_batch_size, device=device)
        mask = mask * logits_mask # Rimuove l'identità dai positivi

        # 5. Calculate Log-Probabilità
        exp_logits = torch.exp(logits) * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)

        # 6. Avg positive pairs
        pos_per_row = mask.sum(1)
        
        pos_per_row_clamped = torch.clamp(pos_per_row, min=1)
        
        # Avg loss
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_row_clamped

        valid_rows = pos_per_row > 0
        if valid_rows.any():
            loss = -mean_log_prob_pos[valid_rows].mean()
        else:
            loss = -mean_log_prob_pos.mean() # Fallback

        return loss
