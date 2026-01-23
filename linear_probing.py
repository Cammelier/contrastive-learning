"""
Linear Probing per valutare le rappresentazioni apprese dal contrastive learning.
Freeze l'encoder e addestra solo un linear classifier.
OPTIMIZED: GPU Transforms + Mixed Precision (Tesla T4 ready)
"""

import hydra 
import torch 
import torch.nn as nn
import wandb
import kornia.augmentation as K
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 

from src.datasets import prepare_loader
from src.models import SimCLR

# --- GPU TRANSFORMS FOR LINEAR PROBING ---

def get_linear_probe_transforms(device):
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.08, 1.0)), # Standard supervised crop
        K.RandomHorizontalFlip(p=0.5),
        K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                    std=torch.tensor([0.247, 0.243, 0.261]))
    ).to(device)

    test_aug = nn.Sequential(
        # In test usually we just CenterCrop or Resize, but for 96x96 we often just Normalize
        # If your images are bigger, add K.CenterCrop((96,96)) here
        K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                    std=torch.tensor([0.247, 0.243, 0.261]))
    ).to(device)
    
    return train_aug, test_aug

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    # Performance Tuning
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 2. Carica checkpoint del modello pretrained
    checkpoint_path = cfg.get('checkpoint_path', None)
    
    # Auto-find logic
    if checkpoint_path is None and cfg.auto_last_checkpoint:
        ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
        # Exclude existing probe checkpoints to avoid confusion
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
    # NOTA: Assumiamo che prepare_loader restituisca immagini "pulite" (solo ToTensor)
    print("Loading data...")
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    
    # Setup GPU Transforms
    train_aug, test_aug = get_linear_probe_transforms(device)
    
    # 5. Carica encoder pretrained e FREEZALO
    print("Loading pretrained encoder...")
    encoder = SimCLR(
        base_model=cfg.model.backbone, 
        out_dim=cfg.model.out_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights (handle potential DataParallel keys if present)
    state_dict = checkpoint['model_state_dict']
    # if 'module.' in list(state_dict.keys())[0]: ... logic to remove module. prefix if needed
    
    encoder.load_state_dict(state_dict, strict=False)
    
    # FREEZE encoder - non verrà addestrato
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    print("✓ Encoder frozen (no gradient updates)")
    
    # 6. Linear classifier
    # Calculate feature dim automatically based on backbone
    if cfg.model.backbone == 'resnet18':
        feature_dim = 512
    elif cfg.model.backbone == 'resnet50':
        feature_dim = 2048
    else:
        feature_dim = 512 # Fallback
        
    num_classes = cfg.data.num_classes
    linear_classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    print(f"Linear classifier: {feature_dim} -> {num_classes}")
    
    # 7. Optimizer e Loss
    optimizer = torch.optim.Adam(
        linear_classifier.parameters(),
        lr=cfg.get('linear_probe_lr', 0.001)
    )
    criterion = nn.CrossEntropyLoss()
    
    # SCALER for Mixed Precision
    scaler = torch.amp.GradScaler()
    
    # 8. Training loop
    best_test_acc = 0.0
    linear_probe_epochs = cfg.get('linear_probe_epochs', 100)
    
    print(f"\n--- Starting Linear Probing ({linear_probe_epochs} epochs) ---\n")
    
    for epoch in range(linear_probe_epochs):
        # TRAIN
        linear_classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{linear_probe_epochs}")
        
        for imgs, labels in pbar:

            # Handle list input (if coming from Contrastive Dataset)
            if isinstance(imgs, list):
                imgs = imgs[0] # Take first view
                
            # Move to GPU
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

                
            # --- GPU AUGMENTATION (Train) ---
            # Apply crop/flip/normalize on GPU
            with torch.no_grad():
                imgs = train_aug(imgs)
            
            # --- FORWARD PASS (Mixed Precision) ---
            with torch.cuda.amp.autocast():
                # Extract features (No grad for encoder)
                with torch.no_grad():
                    features = encoder(imgs, return_features=True)
                
                # Forward classifier
                logits = linear_classifier(features)
                loss = criterion(logits, labels)
            
            # --- BACKWARD (Scaler) ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            current_acc = train_correct / train_total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # TEST (evaluation)
        linear_classifier.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if isinstance(imgs, list):
                    imgs = imgs[0]
                
                # --- GPU AUGMENTATION (Test) ---
                # Normalize only
                imgs = test_aug(imgs)
                
                with torch.cuda.amp.autocast():
                    features = encoder(imgs, return_features=True)
                    logits = linear_classifier(features)
                    loss = criterion(logits, labels)
                
                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        
        # Log & Save
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/accuracy": train_accuracy,
            "test/loss": avg_test_loss,
            "test/accuracy": test_accuracy,
        })
        
        print(f"Epoch [{epoch+1}/{linear_probe_epochs}] | "
              f"Train Acc: {train_accuracy:.4f} | "
              f"Test Acc: {test_accuracy:.4f}")
        
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_model_path = root_dir / "checkpoints" / cfg.experiment.mode / "best_linear_probe.pth"
            torch.save({
                'epoch': epoch + 1,
                'classifier_state_dict': linear_classifier.state_dict(),
                'test_accuracy': test_accuracy,
            }, best_model_path)
    
    # --- Final Evaluation ---
    print(f"\n--- Final Evaluation ---")
    linear_classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Eval"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if isinstance(imgs, list): imgs = imgs[0]
            
            imgs = test_aug(imgs) # Normalize
            
            with torch.cuda.amp.autocast():
                features = encoder(imgs, return_features=True)
                logits = linear_classifier(features)
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Per-class accuracy & Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix (Acc: {best_test_acc:.4f})')
    
    wandb.log({
        "final/best_test_accuracy": best_test_acc,
        "final/confusion_matrix": wandb.Image(fig),
    })
    
    plt.close()
    wandb.finish()

if __name__ == "__main__":
    main()