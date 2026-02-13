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
from src.common.logging import setup_logger

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
def run_validation(model, classifier, val_loader, criterion, device, cfg, class_weights=None):
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for x, labels in val_loader:
            x, labels = x.to(device), labels.to(device).long()
            
            with autocast(device_type='cuda', dtype=dtype):
                # Feature extraction on clean data
                h, _ = model(x)
                logits = classifier(h)
                
                # Cross Entropy with weights for balanced evaluation
                loss = F.cross_entropy(logits, labels, weight=class_weights)
                val_loss += loss.item()
                
                # Accuracy calculation
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)
    
    return val_loss / len(val_loader), val_correct / val_samples if val_samples > 0 else 0



# --- TRAINING MAIN ---
def run_training(cfg: DictConfig, device, model, ckpt_dir: Path):
    # 1. DATA
    train_loader, _ = prepare_loader(cfg, 'train')
    val_loader, class_names = prepare_loader(cfg, 'val')
    
    # --- CLASS WEIGHTS CALCULATION ---
    labels_all = train_loader.dataset.labels
    class_sample_count = np.bincount(labels_all)
    # Inverse frequency weighting
    weights = 1. / class_sample_count
    # Normalizing weights (sum equals number of classes)
    weights = weights / weights.sum() * len(class_sample_count)
    class_weights = torch.FloatTensor(weights).to(device)
    print(f"Class weights applied to Loss: {class_weights.cpu().numpy()}")
    # ---------------------------------

    num_classes = len(class_sample_count)
    
    # 3. OPTIMIZER & SCHEDULER
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.experiment.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.experiment.learning_rate, 
        epochs=cfg.experiment.epochs, steps_per_epoch=len(train_loader)
    )
    
    # 5. CRITERION & LINEAR PROBE
    criterion = ContrastiveLoss(temperature=cfg.experiment.temperature).to(device)
    scaler = GradScaler('cuda')
    
    feat_dim = cfg.model.hidden_dim
    classifier = nn.Sequential(
        nn.BatchNorm1d(feat_dim),
        nn.Linear(feat_dim, num_classes)
    ).to(device)
    cls_optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9)
    
    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        for x, labels in pbar:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(); cls_optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                x_i, x_j = netflow_aug(x)
                h_combined, z_combined = model(torch.cat([x_i, x_j], dim=0))
                
                # 1. Contrastive Loss
                loss_cont = criterion(z_combined)
                
                # 2. Weighted Linear Probing Loss
                h_i = h_combined[:x.size(0)]
                logits = classifier(h_i)
                loss_cls = F.cross_entropy(logits, labels, weight=class_weights)
                
                total_loss = loss_cont + 0.5 * loss_cls
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer); scaler.step(cls_optimizer)
            scaler.update(); scheduler.step()
            pbar.set_postfix(loss=f'{total_loss.item():.3f}')
        
        # Validation with weights
        val_loss, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg, class_weights)
        wandb.log({'epoch': epoch, 'val/acc': val_acc, 'val/loss': val_loss})
        print(f"Epoch {epoch}: Val Acc {val_acc:.2%}")
    
    torch.save({
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'class_names': class_names,
        'class_weights': class_weights.cpu()
    }, ckpt_dir / 'last_model.pth')


#---  TESTING  ---
def run_testing(cfg: DictConfig, device, model, ckpt_dir: Path):
    checkpoint = torch.load(ckpt_dir / 'last_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    class_names = checkpoint.get('class_names', ['Benign', 'Attack'])
    num_classes = len(class_names)
    
    classifier = nn.Sequential(
        nn.BatchNorm1d(cfg.model.hidden_dim),
        nn.Linear(cfg.model.hidden_dim, num_classes)
    ).to(device)
    classifier.load_state_dict(checkpoint['classifier'])
    
    test_loader, _ = prepare_loader(cfg, 'test')
    model.eval(); classifier.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc='Testing'):
            x, labels = x.to(device), labels.to(device).long()
            h, _ = model(x)
            preds = classifier(h).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n[TEST RESULT] Accuracy: {acc:.2f}%")
    
    # LOG CONFUSION MATRIX TO WANDB
    wandb.log({
        "test/acc": acc,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=all_labels, preds=all_preds, class_names=class_names
        )
    })
    return acc



setup_logger() 

@hydra.main(version_base="1.2", config_path="config", config_name="config")
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
