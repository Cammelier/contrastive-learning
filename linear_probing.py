"""
Linear Probing EVALUATION - Optimized for RTX 4090 (Ada Lovelace)
Protocol: SGD + Momentum + BF16 + TF32
"""

import hydra 
import torch 
import torch.nn as nn
import wandb
import kornia.augmentation as K
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sns_setup # alias per seaborn se preferisci, o usa seaborn standard
import seaborn as sns

from src.datasets import prepare_loader
from src.models import SimCLR

# --- Ottimizzazione RTX 4090 (TF32) ---
torch.set_float32_matmul_precision('high')

def get_linear_probe_transforms(device):
    mean = torch.tensor([0.485, 0.456, 0.406]) # Standard ImageNet stats
    std = torch.tensor([0.229, 0.224, 0.225])
    
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.5, 1.0)), 
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(mean=mean, std=std)
    ).to(device)

    test_aug = nn.Sequential(
        K.Normalize(mean=mean, std=std)
    ).to(device)
    
    return train_aug, test_aug

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup Ambiente
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    # 2. Checkpoint Logic
    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None and cfg.get('auto_last_checkpoint', False):
        ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
        checkpoints = [p for p in ckpt_dir.glob("*.pth") if "probe" not in p.name]
        checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else None

    # 3. Wandb
    wandb.init(
        project=cfg.logger.project,
        group=f"{cfg.experiment.mode}_linear_probing",
        name=f"probe_4090_bf16_lr{cfg.get('linear_probe_lr', 0.1)}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 4. Data (Aumenta num_workers nel config per la 4090!)
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    train_aug, test_aug = get_linear_probe_transforms(device)
    
    # 5. Encoder Frozen
    encoder = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # 6. Linear Classifier con Normalizzazione (MoCo style)
    feature_dim = 512 if cfg.model.backbone == 'resnet18' else 2048
    linear_classifier = nn.Sequential(
        nn.BatchNorm1d(feature_dim, affine=False), 
        nn.Linear(feature_dim, cfg.data.num_classes)
    ).to(device)
    
    # 7. Optimizer (SGD + Momentum) e Mixed Precision
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=cfg.get('linear_probe_lr', 0.1), # Con 4090 puoi testare anche 0.3 o 0.5
        momentum=0.9,
        weight_decay=0
    )
    
    epochs = cfg.get('linear_probe_epochs', 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # RTX 4090 supporta BFloat16 nativamente
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16))
    criterion = nn.CrossEntropyLoss()

    # 8. Loop
    for epoch in range(epochs):
        linear_classifier.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [BF16]")
        
        for imgs, labels in pbar:
            if isinstance(imgs, list): imgs = imgs[0]
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.no_grad(): imgs = train_aug(imgs)
            
            optimizer.zero_grad(set_to_none=True) # Ottimizzazione memoria 4090
            
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                with torch.no_grad():
                    features = encoder(imgs, return_features=True)
                logits = linear_classifier(features)
                loss = criterion(logits, labels)
            
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'acc': f'{train_correct/train_total:.4f}'})
        
        scheduler.step()
        
        # Test
        linear_classifier.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                if isinstance(imgs, list): imgs = imgs[0]
                imgs = test_aug(imgs.to(device))
                with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                    features = encoder(imgs, return_features=True)
                    logits = linear_classifier(features)
                test_correct += (logits.argmax(1) == labels.to(device)).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        wandb.log({"epoch": epoch + 1, "test/acc": test_acc, "lr": optimizer.param_groups[0]['lr']})

    # --- Final Confusion Matrix ---
    linear_classifier.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            if isinstance(imgs, list): imgs = imgs[0]
            imgs = test_aug(imgs.to(device))
            logits = linear_classifier(encoder(imgs, return_features=True))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    wandb.log({"final/cm": wandb.Image(plt)})
    wandb.finish()

if __name__ == "__main__":
    main()

