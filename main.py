import hydra 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 
from torch.amp import autocast, GradScaler

from src.datasets import prepare_loader
from src.models import SimCLR
from src.losses import ContrastiveLoss

from src.plot import generate_tsne_plot

import warnings
warnings.filterwarnings("ignore")

# --- NETFLOW AUGMENTATIONS ---
def netflow_aug(x, noise_strength=0.02, drop_prob=0.1):
    """Genera 2 viste per vettori NetFlow [B, features]"""
    noise = noise_strength * torch.randn_like(x)
    
    # Vista 1: 10% feature dropout + noise
    mask1 = torch.rand_like(x) > drop_prob
    x_i = x * mask1 + noise * (~mask1)
    
    # Vista 2: 15% diverso + noise
    mask2 = torch.rand_like(x) > (drop_prob + 0.05)
    x_j = x * mask2 + noise * (~mask2)
    
    return x_i, x_j

# --- VALIDATION (NetFlow) ---
def run_validation(engine, val_loader, device, cfg):
    engine.model.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for x, labels in val_loader:
            x, labels = x.to(device), labels.to(device).long()
            
            with autocast(device_type='cuda', dtype=dtype):
                x_i, x_j = netflow_aug(x)
                _, z_combined = engine.model(torch.cat([x_i, x_j], dim=0))
                loss = engine.criterion(z_combined, labels if cfg.experiment.supervised else None)
                val_loss += loss.item()
                
                # Linear probing acc
                if labels.min() >= 0:
                    h_i = engine.model.backbone(torch.cat([x_i, x_j], dim=0))[:x.size(0)]
                    logits = engine.classifier(h_i)
                    preds = logits.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += labels.size(0)
    
    return val_loss / len(val_loader), val_correct / val_samples if val_samples > 0 else 0

# --- TRAINING MAIN ---
def run_training(cfg: DictConfig, device, model,ckpt_dir: Path):
    # 1. DATA
    train_loader, _ = prepare_loader(cfg, 'train' if cfg.experiment.supervised else 'unlabeled')
    val_loader, class_names = prepare_loader(cfg, 'val')
    
    # 2. MODEL
    model = SimCLR(
        input_dim=cfg.data.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim,
        num_classes=len(class_names) if cfg.experiment.supervised else None
    ).to(device)
    
    # 3. OPTIMIZER AdamW 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.experiment.learning_rate,
        weight_decay=cfg.experiment.weight_decay,
        betas=(0.9, 0.99)
    )
    
    # 4. SCHEDULER
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.experiment.learning_rate,
        epochs=cfg.experiment.epochs,
        steps_per_epoch=len(train_loader)
    )
    
    # 5. CRITERION
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=cfg.experiment.supervised
    ).to(device)
    
    scaler = GradScaler('cuda')
    
    # 6. LINEAR PROBE CLASSIFIER
    feat_dim = cfg.model.hidden_dim
    classifier = nn.Sequential(
        nn.BatchNorm1d(feat_dim),
        nn.Linear(feat_dim, len(class_names))
    ).to(device)
    cls_optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9)
    
    # TRAINING LOOP
    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        for x, labels in pbar:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()
            
            optimizer.zero_grad()
            cls_optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                
                x_i, x_j = netflow_aug(x)
                h_combined, z_combined = model(torch.cat([x_i, x_j], dim=0))
                
                # Contrastive Loss
                loss = criterion(z_combined, labels if cfg.experiment.supervised else None)
                
                # Linear probing loss
                if labels.min() >= 0:
                    h_i = h_combined[:x.size(0)]
                    logits = classifier(h_i)
                    cls_loss = F.cross_entropy(logits, labels)
                    total_loss = loss + 0.1 * cls_loss  # weighted
                else:
                    total_loss = loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.step(cls_optimizer)
            scaler.update()
            
            scheduler.step()
            pbar.set_postfix(loss=f'{total_loss.item():.3f}')
        
        # Validation
        val_loss, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg)
        wandb.log({
            'epoch': epoch, 'train/loss': total_loss.item(),
            'val/loss': val_loss, 'val/acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        print(f"Epoch {epoch}: Val Acc {val_acc:.2%}")
    
    # Save
    torch.save({
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'class_names': class_names
    }, ckpt_dir / 'last_model.pth')

# --- TESTING ---
def run_testing(cfg: DictConfig, device, model, ckpt_dir: Path):
    checkpoint = torch.load(ckpt_dir / 'last_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    test_loader, class_names = prepare_loader(cfg, 'test')
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc='Test'):
            x, labels = x.to(device), labels.to(device)
            x_i, x_j = netflow_aug(x)
            h = model.backbone(torch.cat([x_i, x_j], dim=0))[:x.size(0)]
            logits = model.classifier(h)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = 100 * correct / total
    print(f"TEST ACC: {acc:.2f}%")
    wandb.log({'test/acc': acc})
    return acc

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir / f"checkpoints/{cfg.experiment.mode}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    
    wandb.init(project=cfg.logger.wandb.project, config=OmegaConf.to_container(cfg))
    
    model = SimCLR(
        input_dim=cfg.data.input_dim,  
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim
    ).to(device)
    
    stage = cfg.get('stage', 'all')
    if stage in ['all', 'training']:
        run_training(cfg, device, model, ckpt_dir)
    if stage in ['all', 'testing']:
        run_testing(cfg, device, model, ckpt_dir)
    
    wandb.finish()

if __name__ == "__main__":
    main()
