"""
Linear Probing EVALUATION - NetFlow Optimized (RTX 4090 Ada Lovelace)
Protocol: SGD + Momentum + BF16 + TF32 + Tabular Augs
"""

import hydra 
import torch 
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.datasets import prepare_loader
from src.models import SimCLR

# TF32 optimization
torch.set_float32_matmul_precision('high')

# --- NETFLOW AUGS per Linear Probing ---
def netflow_probe_aug(x, noise_strength=0.01):
    """Light augmentations per linear probing (meno aggressive)"""
    noise = noise_strength * torch.randn_like(x)
    mask = torch.rand_like(x) > 0.05  # solo 5% dropout
    return x * mask + noise * (~mask)

# --- LINEAR PROBE TRANSFORMS ---
def get_netflow_probe_transforms(device):
    return netflow_probe_aug  

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    # Checkpoint (self-supervised pretraining)
    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None and cfg.get('auto_last_checkpoint', False):
        ckpt_dir = root_dir / "checkpoints" / "self_supervised"
        checkpoints = list(ckpt_dir.glob("*.pth"))
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None
    
    wandb.init(
        project=cfg.logger.project,
        group=f"{cfg.experiment.mode}_netflow_linear_probe",
        name=f"netflow_probe_bf16_lr{cfg.linear_probe_lr}",
        config=OmegaConf.to_container(cfg)
    )
    
    # DATA: NetFlow
    train_loader, _ = prepare_loader(cfg, split='train')
    test_loader, class_names = prepare_loader(cfg, split='test')
    
    # ENCODER Frozen (NetFlow MLP)
    encoder = SimCLR(
        input_dim=cfg.data.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim
    ).to(device)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['model'], strict=False)
    
    # FREEZE encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # LINEAR CLASSIFIER 
    feature_dim = cfg.model.hidden_dim  
    linear_classifier = nn.Sequential(
        nn.BatchNorm1d(feature_dim, affine=False),
        nn.Linear(feature_dim, len(class_names))
    ).to(device)
    
    
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=cfg.linear_probe_lr,  
        momentum=0.9,
        weight_decay=0
    )
    
    epochs = cfg.linear_probe_epochs  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # BF16 per 4090
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16))
    criterion = nn.CrossEntropyLoss()
    
    # TRAINING LOOP
    for epoch in range(epochs):
        linear_classifier.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{epochs} [BF16]")
        
        for x, labels in pbar:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Light aug per probing
            x_aug = netflow_probe_aug(x)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                with torch.no_grad():
                    features = encoder.backbone(x_aug)  
                logits = linear_classifier(features)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(acc=f'{train_correct/train_total:.4f}')
        
        scheduler.step()
        
        # VALIDATION
        linear_classifier.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, labels in test_loader:
                x_aug = netflow_probe_aug(x.to(device))
                with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                    features = encoder.backbone(x_aug)
                    logits = linear_classifier(features)
                test_correct += (logits.argmax(1) == labels.to(device)).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        wandb.log({"epoch": epoch+1, "test/acc": test_acc, "lr": optimizer.param_groups[0]['lr']})
    
    # FINAL CONFUSION MATRIX
    linear_classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x_aug = netflow_probe_aug(x.to(device))
            logits = linear_classifier(encoder.backbone(x_aug))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    wandb.log({"final_cm": wandb.Image(plt)})
    plt.close()
    wandb.finish()
    
    print(f"✓ Linear Probe Final Acc: {test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
