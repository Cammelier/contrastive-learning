import torch 
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, supervised=False):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.supervised = supervised

    def forward(self, features, labels=None):
        """
        Args: 
            features: Tensor of shape (full_batch_size, feature_dim).
                     In SimCLR, full_batch_size is usually 2 * batch_size.
            labels: Optional Tensor of shape (batch_size,) or (full_batch_size,).
        """
        device = features.device
        
        # 1. L2 Normalization (Crucial for cosine similarity)
        features = F.normalize(features, dim=1)
        
        full_batch_size = features.shape[0]

        # 2. Compute Logits (Similarity matrix scaled by temperature)
        # Resulting shape: (full_batch_size x full_batch_size)
        logits = torch.matmul(features, features.T) / self.temperature

        # For numerical stability: subtract the maximum logit to avoid overflow in exp()
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 3. Create Positive Mask
        if labels is None:
            # Self-supervised case (SimCLR)
            # Each image i is positive only with its augmented version j
            batch_size = full_batch_size // 2
            mask = torch.eye(batch_size, device=device).repeat(2, 2)
        else: 
            # Supervised case (SupCon)
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != full_batch_size:
                # If labels are provided for half batch only, repeat them to align
                labels = labels.repeat(2, 1)
            
            # Matrix of 1s where labels match, 0s otherwise
            mask = torch.eq(labels, labels.T).float().to(device)
            
        # 4. Auto-similarity exclusion (Diagonal of the matrix)
        # We must prevent an image from being used as its own positive or negative example
        logits_mask = torch.scatter(
            torch.ones_like(mask, device=device), 
            1, 
            torch.arange(full_batch_size, device=device).view(-1, 1), 
            0 
        )
        
        # Apply the mask to both the similarity matrix and the positive pairs mask
        mask = mask * logits_mask

        # 5. Compute Log-Probability
        # Denominator: sum of exponentials of logits, excluding self-similarity
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = log( exp(zi*zj/tau) / sum(exp(zi*zk/tau)) )
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)

        # 6. Average over positive pairs
        # For each row, average the log-probabilities of all valid positive samples
        pos_per_row = mask.sum(1)
        
        # Avoid division by zero if a class appears only once in the batch
        pos_per_row = torch.where(pos_per_row > 0, pos_per_row, torch.ones_like(pos_per_row, device=device))
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_row

        # The loss is the negative mean of the log-probabilities
        return -mean_log_prob_pos.mean()
