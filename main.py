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
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score
from src.datasets import prepare_loader
from src.common.plot import plot_enhanced_confusion_matrix 

import warnings
warnings.filterwarnings("ignore")

# --- NETFLOW AUGMENTATIONS ---
def netflow_aug(x, noise_strength=0.02, drop_prob=0.1):
    batch_size, num_features = x.shape
    device = x.device

    # VIEW 1: Noise + Masking
    noise1 = noise_strength * torch.randn_like(x)
    mask1 = torch.rand_like(x) > drop_prob
    x_i = x * mask1 + noise1
    
    # VIEW 2: Swap Augmentation (Sostituzione parziale da altri campioni nel batch)
    random_indices = torch.randperm(batch_size).to(device)
    x_random = x[random_indices] 
    swap_mask = torch.rand_like(x) < 0.15 
    x_j = torch.where(swap_mask, x_random, x) 
    x_j = x_j + noise_strength * torch.randn_like(x_j)

    return x_i, x_j

# --- CONTRASTIVE PRETRAINING ---
def run_contrastive_pretraining(cfg: DictConfig, device, model, ckpt_dir: Path):
    train_loader, _, _ = prepare_loader(cfg, 'train')
    supervised = cfg.experiment.get('supervised', False)
    mode_name = "SupCon" if supervised else "SimCLR"
    
    print(f"\n🚀 Starting {mode_name} Pretraining")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.experiment.learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.experiment.learning_rate, 
        epochs=cfg.experiment.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    from src.losses import ContrastiveLoss # Assicurati che sia importabile
    criterion = ContrastiveLoss(temperature=cfg.experiment.temperature, supervised=supervised).to(device)
    scaler = GradScaler('cuda')
    best_loss = float('inf')

    for epoch in range(cfg.experiment.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}")
        
        for x, labels in pbar:
            x, labels = x.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                x_i, x_j = netflow_aug(x)
                h_combined, z_combined = model(torch.cat([x_i, x_j], dim=0))
                
                if supervised:
                    labels_combined = torch.cat([labels, labels], dim=0)
                    loss = criterion(z_combined, labels_combined)
                else:
                    loss = criterion(z_combined)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({'pretrain/loss': avg_loss, 'pretrain/epoch': epoch+1})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model': model.state_dict(), 'mode': mode_name}, ckpt_dir / 'pretrained_encoder.pth')

# --- LINEAR PROBING ---
def run_linear_probe(cfg: DictConfig, device, model, ckpt_dir: Path, loss_weights: torch.Tensor):
    print(f"\n🔬 Linear Probing (Encoder Frozen)")
    train_loader, class_names, _ = prepare_loader(cfg, 'train')
    val_loader, _, _ = prepare_loader(cfg, 'val')
    
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    
    classifier = nn.Linear(cfg.model.hidden_dim, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg.experiment.get('probe_lr', 1e-3))
    
    best_f1 = 0.0
    for epoch in range(cfg.experiment.get('probe_epochs', 20)):
        classifier.train()
        for x, labels in tqdm(train_loader, desc=f"Probe Epoch {epoch+1}"):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad(): features, _ = model(x)
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                features, _ = model(x)
                all_preds.extend(classifier(features).argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        b_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f"Probe F1: {f1:.4f} | B-Acc: {b_acc:.2%}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'encoder': model.state_dict(), 'classifier': classifier.state_dict(), 'class_names': class_names}, ckpt_dir / 'best_linear_probe.pth')
            
    return best_f1

# --- FINE-TUNING ---
def run_fine_tuning(cfg: DictConfig, device, model, ckpt_dir: Path, loss_weights: torch.Tensor):
    print(f"\n🎯 Fine-Tuning (End-to-End)")
    train_loader, class_names, _ = prepare_loader(cfg, 'train')
    val_loader, _, _ = prepare_loader(cfg, 'val')
    
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.classifier = nn.Linear(cfg.model.hidden_dim, len(class_names)).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.experiment.get('ft_lr_backbone', 1e-6)},
        {'params': model.classifier.parameters(), 'lr': cfg.experiment.get('ft_lr_head', 1e-3)}
    ])
    
    best_f1 = 0.0
    scaler = GradScaler('cuda')
    
    for epoch in range(cfg.experiment.get('ft_epochs', 10)):
        model.train()
        for x, labels in tqdm(train_loader, desc=f"FT Epoch {epoch+1}"):
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                features, _ = model(x)
                loss = criterion(model.classifier(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                f, _ = model(x)
                all_preds.extend(model.classifier(f).argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"FT F1 Macro: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model': model.state_dict(), 'class_names': class_names}, ckpt_dir / 'best_finetuned.pth')
    return best_f1

# --- TESTING ---
def run_testing(cfg, device, model, ckpt_dir: Path, mode='linear_probe'):
    checkpoint_path = ckpt_dir / ('best_linear_probe.pth' if mode == 'linear_probe' else 'best_finetuned.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    
    if mode == 'linear_probe':
        model.load_state_dict(checkpoint['encoder'], strict=False)
        classifier = nn.Linear(cfg.model.hidden_dim, len(class_names)).to(device)
        classifier.load_state_dict(checkpoint['classifier'])
    else:
        model.load_state_dict(checkpoint['model'])
        classifier = model.classifier

    test_loader, _ , _ = prepare_loader(cfg, 'test')
    model.eval(); classifier.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc=f"Testing {mode}"):
            f, _ = model(x.to(device))
            all_preds.extend(classifier(f).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    plot_enhanced_confusion_matrix(all_labels, all_preds, class_names, mode)
    print(f"\n📊 REPORT {mode.upper()}:\n", classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    wandb.log({f"test_{mode}_f1": f1_score(all_labels, all_preds, average='macro')})

# --- MAIN ---
@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    supervised = cfg.experiment.get('supervised', False)
    mode_name = "supcon" if supervised else "simclr"
    ckpt_dir = root_dir / f"checkpoints/{mode_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project=cfg.logger.wandb.project, name=f"{mode_name}_{cfg.data.file_name}", config=OmegaConf.to_container(cfg))
    
    from src.models import SimCLR # Assicurati che SimCLR accetti questi parametri
    model = SimCLR(input_dim=cfg.data.input_dim, hidden_dim=cfg.model.hidden_dim, out_dim=cfg.model.out_dim).to(device)
    
    # 1. Carica pesi loss una volta sola
    _, _, loss_weights = prepare_loader(cfg, 'train')
    
    stage = cfg.get('stage', 'all')
    if stage in ['all', 'pretrain']: run_contrastive_pretraining(cfg, device, model, ckpt_dir)
    if stage in ['all', 'probe']: run_linear_probe(cfg, device, model, ckpt_dir, loss_weights)
    if stage in ['all', 'finetune']: run_fine_tuning(cfg, device, model, ckpt_dir, loss_weights)
    if stage in ['all', 'test']:
        run_testing(cfg, device, model, ckpt_dir, mode='linear_probe')
        run_testing(cfg, device, model, ckpt_dir, mode='finetuned')
    
    wandb.finish()

if __name__ == "__main__":
    main()
