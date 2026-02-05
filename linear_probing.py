"""
Linear Probing per valutare le rappresentazioni apprese dal contrastive learning.
Freeze l'encoder e addestra solo un linear classifier.
OPTIMIZED: CPU/GPU Auto-detection + Mixed Precision (Tesla T4 ready)
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
import seaborn as sns

from src.datasets import prepare_loader
from src.models import SimCLR

# --- GPU/CPU TRANSFORMS FOR LINEAR PROBING ---
def get_linear_probe_transforms(device):
    mean = torch.tensor([0.4467, 0.4398, 0.4066])
    std = torch.tensor([0.2603, 0.2566, 0.2713])
    
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
    # 1. Setup
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 2. Carica checkpoint del modello pretrained
    checkpoint_path = cfg.get('checkpoint_path', None)
    
    if checkpoint_path is None and cfg.get('auto_last_checkpoint', False):
        ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
        checkpoints = [p for p in ckpt_dir.glob("*.pth") if "probe" not in p.name]
        
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        else:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    
    # 3. Wandb setup
    wandb.init(
        project=cfg.logger.project,
        group=f"{cfg.experiment.mode}_linear_probing",
        name=f"probe_{cfg.model.backbone}_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 4. Carica dataloaders
    print("Loading data...")
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    
    train_aug, test_aug = get_linear_probe_transforms(device)
    
    # 5. Carica encoder pretrained e FREEZALO
    print("Loading pretrained encoder...")
    encoder = SimCLR(
        base_model=cfg.model.backbone, 
        out_dim=cfg.model.out_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    encoder.load_state_dict(state_dict, strict=False)
    
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    print("✓ Encoder frozen (no gradient updates)")
    
    # 6. Linear classifier
    feature_dim = 512 if cfg.model.backbone == 'resnet18' else 2048
    num_classes = cfg.data.num_classes
    linear_classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    print(f"Linear classifier: {feature_dim} -> {num_classes}")
    
    # 7. Optimizer e Loss
    optimizer = torch.optim.Adam(
        linear_classifier.parameters(),
        lr=cfg.get('linear_probe_lr', 0.0003),
        weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # 8. Training loop
    best_test_acc = 0.0
    linear_probe_epochs = cfg.get('linear_probe_epochs', 20)
    
    print(f"\n--- Starting Linear Probing ({linear_probe_epochs} epochs) ---\n")
    
    for epoch in range(linear_probe_epochs):
        linear_classifier.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{linear_probe_epochs}")
        
        for imgs, labels in pbar:
            if isinstance(imgs, list): imgs = imgs[0]
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.no_grad(): imgs = train_aug(imgs)
            
            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    features = encoder(imgs, return_features=True)
                logits = linear_classifier(features)
                loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({'acc': f'{train_correct/train_total:.4f}'})
        
        # Test (evaluation)
        linear_classifier.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
                if isinstance(imgs, list): imgs = imgs[0]
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                imgs = test_aug(imgs)
                
                with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                    features = encoder(imgs, return_features=True)
                    logits = linear_classifier(features)
                    loss = criterion(logits, labels)
                
                test_loss += loss.item()
                test_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                test_total += labels.size(0)
        
        test_accuracy = test_correct / test_total
        wandb.log({"epoch": epoch + 1, "train/accuracy": train_correct/train_total, "test/accuracy": test_accuracy})
        
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_model_path = root_dir / "checkpoints" / cfg.experiment.mode / "best_linear_probe.pth"
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch + 1, 'classifier_state_dict': linear_classifier.state_dict(), 'test_accuracy': test_accuracy}, best_model_path)
    
    # --- Final Evaluation ---
    print(f"\n--- Final Evaluation & Confusion Matrix ---")
    linear_classifier.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Eval"):
            if isinstance(imgs, list): imgs = imgs[0]
            imgs = test_aug(imgs.to(device))
            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                features = encoder(imgs, return_features=True)
                logits = linear_classifier(features)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else test_loader.dataset.dataset.classes
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'Confusion Matrix (Acc: {best_test_acc:.4f})')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    wandb.log({"final/confusion_matrix": wandb.Image(fig)})
    wandb.finish()

if __name__ == "__main__":
    main()
