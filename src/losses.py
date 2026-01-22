def forward(self, features, labels=None):
    device = features.device
    features = F.normalize(features, dim=1)

    logits = torch.matmul(features, features.T) / self.temperature
    
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    batch_size = features.shape[0]
    logits_mask = torch.ones_like(logits) - torch.eye(batch_size, device=device)

    if labels is None:
        # SELF-SUPERVISED (SimCLR)
        half_batch = batch_size // 2
        mask = torch.zeros_like(logits)
        mask[torch.arange(half_batch), torch.arange(half_batch, batch_size)] = 1
        mask[torch.arange(half_batch, batch_size), torch.arange(half_batch)] = 1
    else:
        #  SUPERVISED (SupCon)
        labels = labels.view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float().to(device)
        
        mask = mask * logits_mask

 
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-06)

   
    pos_counts = mask.sum(1)
    pos_counts[pos_counts == 0] = 1 
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / pos_counts

    return -mean_log_prob_pos.mean()
