"""
Fine-Tuning Script - NetFlow Optimized (RTX 4090 Ada Lovelace)
Protocol: Differential LR + BF16 + TF32 + AdamW + Tabular Augs
"""

import hydra 
import torch 
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.datasets import prepare_loader
from src.models import NetflowSimCLR

# TF32 optimization
torch.set_float32_matmul_precision('high')

# --- NETFLOW AUGS per Fine-Tuning ---
def netflow_ft_aug(x, noise_strength=0.015, drop_prob=0.08):
    """Moderate augmentations per fine-tuning (bilanciate)"""
    noise = noise_strength * torch.randn_like(x)
    mask1 = torch.rand_like(x) > drop_prob
    mask2 = torch.rand_like(x) > (drop_prob + 0.02)
    x1 = x * mask1 + noise * (~mask1)
    x2 = x * mask2 + noise * (~mask2)
    return x1, x2  

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    wandb.init(
        project=cfg.logger.project, 
        group="netflow_finetuning_4090", 
        name=f"netflow_ft_bf16_{cfg.data.name}",
        config=OmegaConf.to_container(cfg)
    )
    
    # DATA NetFlow
    train_loader = prepare_loader(cfg, split='train')
    test_loader, class_names = prepare_loader(cfg, split='test')
    
    # MODEL: load pretraining self-supervised
    model = NetflowSimCLR(
        input_dim=cfg.data.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_classes=len(class_names)
    ).to(device)
    
    # Load self-supervised weights
    ckpt_path = root_dir / "checkpoints" / "self_supervised" / "last_model.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"✓ Loaded pretraining: {ckpt_path}")
    
    # Classifier head 
    model.classifier = nn.Linear(cfg.model.hidden_dim, len(class_names)).to(device)
    
    # DIFFERENTIAL LR: backbone lento, head veloce
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.ft_lr_backbone},  # 1e-5
        {'params': model.classifier.parameters(), 'lr': cfg.ft_lr_head}     # 1e-4
    ], weight_decay=cfg.weight_decay)
    
    # BF16 RTX 4090
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # TRAINING
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{cfg.epochs} [BF16]")
        
        for x, labels in pbar:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Augmentations NetFlow
            x_aug = netflow_ft_aug(x)[0]  # usa prima vista
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                # Features dal backbone
                features = model.backbone(x_aug)
                logits = model.classifier(features)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(acc=f'{train_correct/train_total:.2%}')
        
        scheduler.step()
        
        # TEST
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, labels in test_loader:
                x_aug = netflow_ft_aug(x.to(device))[0]
                with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                    features = model.backbone(x_aug)
                    logits = model.classifier(features)
                test_correct += (logits.argmax(1) == labels.to(device)).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        wandb.log({"epoch": epoch+1, "test/acc": test_acc})
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 
                      root_dir / "checkpoints" / "finetuning" / f"best_ft_{cfg.data.name}.pth")
    
    # CONFUSION MATRIX
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in test_loader:
            x_aug = netflow_ft_aug(x.to(device))[0]
            logits = model.classifier(model.backbone(x_aug))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    wandb.log({"final_cm": wandb.Image(plt)})
    plt.close()
    wandb.finish()
    
    print(f"✓ Fine-Tuning Best Acc: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
