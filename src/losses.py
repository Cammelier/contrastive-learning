import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None):
        """
        Args: 
            features: Tensor of shape (batch_size, feature_dim)
            labels: Optional Tensor of shape (batch_size,) containing class labels
        """

        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        f1, f2 = torch.unbind(features, dim=1)
        features = torch.cat([f1,f2],dim=0)
        features = F.normalize(features, dim=1)

        # Create positive mask
        if labels is None:
            # Self-supervised case
            mask = torch.eye(batch_size, device=device).repeat(2,2)
        else: 
            # Supervised case
            labels = labels.contiguous().view(-1,1)
            mask = torch.eq(labels, labels.T).float().to(device)
            mask = mask.repeat(2,2)
        
        # Compute logits
        logits = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # auto-similarity esclusion
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size*2,device=device).view(-1,1),0 
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)

        # Average for positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-06)

        return -mean_log_prob_pos.mean()