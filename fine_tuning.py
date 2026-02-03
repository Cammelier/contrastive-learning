"""
Fine-Tuning Script (Optimized against Overfitting).
Includes automatic Confusion Matrix generation with class names.
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
import numpy as np

from src.datasets import prepare_loader
from src.models import SimCLR

# --- STRONG AUGMENTATIONS FOR FINE-TUNING ---
def get_strong_finetuning_transforms(device):
    # Stats for STL-10
    mean = torch.tensor([0.4467, 0.4398, 0.4066])
    std = torch.tensor([0.2603, 0.2566, 0.2713])
    
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.5, 1.0)), 
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
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
    # Rilevamento automatico: usa GPU se disponibile, altrimenti CPU
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    wandb.init(
        project=cfg.logger.project, 
        group="finetuning_strong", 
        name=f"ft_strong_{cfg.model.backbone}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 2. Data Preparation
    print("Loading data...")
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    
    if hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
    else:
        class_names = test_loader.dataset.dataset.classes
    print(f"✓ Detected classes: {class_names}")

    train_aug, test_aug = get_strong_finetuning_transforms(device)

    # 3. Model Setup
    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)
    
    ckpt_path = root_dir / "checkpoints" / "self_supervised" / "last_model.pth"
    if ckpt_path.exists():
        print(f"Loading pre-trained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    feature_dim = 512 if cfg.model.backbone == 'resnet18' else 2048 
    model.classifier = nn.Linear(feature_dim, cfg.data.num_classes).to(device)

    probe_path = root_dir / "checkpoints" / "self_supervised" / "best_linear_probe.pth"
    if probe_path.exists():
        print("✓ Loading Linear Probe weights...")
        probe_ckpt = torch.load(probe_path, map_location=device)
        model.classifier.load_state_dict(probe_ckpt['classifier_state_dict'])

    # 4. FULL UNFREEZE
    for param in model.parameters():
        param.requires_grad = True
    
    # 5. Optimizer, Scheduler & Loss
    ft_epochs = cfg.epochs
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # Lo scaler è utile solo su GPU, su CPU usiamo None
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # 6. Training Loop
    best_acc = 0.0
    print(f"\n--- Starting Fine-Tuning ({ft_epochs} epochs) ---\n")

    for epoch in range(ft_epochs):
        model.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{ft_epochs}")
        
        for imgs, labels in pbar:
            if isinstance(imgs, list): imgs = imgs[0]
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad(): imgs = train_aug(imgs)
            
            # Autocast attivato solo se disponibile CUDA
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                features = torch.flatten(model.backbone(imgs), 1) 
                logits = model.classifier(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'acc': f'{train_correct/train_total:.2%}'})

        # --- VALIDATION ---
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                if isinstance(imgs, list): imgs = imgs[0]
                imgs, labels = imgs.to(device), labels.to(device)
                imgs = test_aug(imgs) 
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model.classifier(torch.flatten(model.backbone(imgs), 1))
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        print(f"Epoch {epoch+1} Result | Train Acc: {train_correct/train_total:.2%} | Test Acc: {test_acc:.2%}")
        
        scheduler.step()
        wandb.log({"epoch": epoch + 1, "ft/train_acc": train_correct/train_total, "ft/test_acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'accuracy': best_acc}, 
                       root_dir / "checkpoints" / "self_supervised" / "finetuned_best.pth")

    # --- 7. FINAL EVALUATION FOR CONFUSION MATRIX ---
    print("\nGenerating Confusion Matrix on Test Set...")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Eval"):
            if isinstance(imgs, list): imgs = imgs[0]
            imgs = test_aug(imgs.to(device))
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model.classifier(torch.flatten(model.backbone(imgs), 1))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {cfg.model.backbone} (Acc: {best_acc:.2%})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    cm_path = root_dir / "checkpoints" / "self_supervised" / "confusion_matrix.png"
    plt.savefig(cm_path)
    wandb.log({"final/confusion_matrix": wandb.Image(str(cm_path))})
    
    wandb.finish()

if __name__ == "__main__":
    main()
