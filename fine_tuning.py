"""
Fine-Tuning Script - Optimized for RTX 4090 (Ada Lovelace)
Protocol: Differential LR + BF16 + TF32 + AdamW
"""

import hydra 
import torch 
import torch.nn as nn
import wandb
import kornia.augmentation as K
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.datasets import prepare_loader
from src.models import SimCLR

# --- Ottimizzazione RTX 4090 (TF32) ---
torch.set_float32_matmul_precision('high')

def get_strong_finetuning_transforms(device):
    mean = torch.tensor([0.4467, 0.4398, 0.4066])
    std = torch.tensor([0.2603, 0.2566, 0.2713])
    
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.8, 1.0)), # Più conservativo per fine-tuning
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5), # Meno aggressivo per non distruggere le feature pre-apprese
        K.Normalize(mean=mean, std=std)
    ).to(device)

    test_aug = nn.Sequential(
        K.Normalize(mean=mean, std=std)
    ).to(device)
    
    return train_aug, test_aug

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup Environment
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    wandb.init(
        project=cfg.logger.project, 
        group="finetuning_4090", 
        name=f"ft_bf16_{cfg.model.backbone}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 2. Data
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else test_loader.dataset.dataset.classes
    train_aug, test_aug = get_strong_finetuning_transforms(device)

    # 3. Model Setup
    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)
    
    # Caricamento pesi pre-trained (SimCLR)
    ckpt_path = root_dir / "checkpoints" / "self_supervised" / "last_model.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Classifier (Linear Layer aggiuntivo)
    feature_dim = 512 if cfg.model.backbone == 'resnet18' else 2048 
    model.classifier = nn.Linear(feature_dim, cfg.data.num_classes).to(device)

    # 4. Optimizer & Mixed Precision (RTX 4090)
    # Differential Learning Rates: l'encoder impara piano, il classifier più veloce
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.get('ft_lr_backbone', 1e-5)},
        {'params': model.classifier.parameters(), 'lr': cfg.get('ft_lr_head', 1e-4)}
    ], weight_decay=cfg.get('weight_decay', 0.05))

    # BFloat16 per la 4090
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Lo scaler serve solo se usi float16, con bfloat16 non è strettamente necessario ma lo teniamo
    scaler = torch.amp.GradScaler('cuda', enabled=(autocast_dtype == torch.float16))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [BF16]")
        
        for imgs, labels in pbar:
            if isinstance(imgs, list): imgs = imgs[0]
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with torch.no_grad(): imgs = train_aug(imgs)
            
            optimizer.zero_grad(set_to_none=True) # Ottimizzato per 4090
            
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                # Importante: usa model.backbone per estrarre le feature
                features = torch.flatten(model.backbone(imgs), 1) 
                logits = model.classifier(features)
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
            pbar.set_postfix({'acc': f'{train_correct/train_total:.2%}'})

        # --- VALIDATION ---
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                if isinstance(imgs, list): imgs = imgs[0]
                imgs = test_aug(imgs.to(device))
                with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                    logits = model.classifier(torch.flatten(model.backbone(imgs), 1))
                test_correct += (logits.argmax(1) == labels.to(device)).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        scheduler.step()
        wandb.log({"epoch": epoch + 1, "test/acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), root_dir / "checkpoints" / "self_supervised" / "finetuned_best.pth")

    # --- 6. FINAL EVALUATION & CONFUSION MATRIX ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            if isinstance(imgs, list): imgs = imgs[0]
            imgs = test_aug(imgs.to(device))
            logits = model.classifier(torch.flatten(model.backbone(imgs), 1))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    wandb.log({"final/cm": wandb.Image(plt)})
    wandb.finish()

if __name__ == "__main__":
    main()
