"""
Fine-Tuning Script (Optimized against Overfitting).
Loads the pre-trained SimCLR encoder, unfreezes all layers, 
and trains with STRONG augmentations and HIGHER weight decay.
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

# --- STRONG AUGMENTATIONS FOR FINE-TUNING ---
def get_strong_finetuning_transforms(device):
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.2, 1.0)), 
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                    std=torch.tensor([0.247, 0.243, 0.261]))
    ).to(device)

   
    test_aug = nn.Sequential(
        K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                    std=torch.tensor([0.247, 0.243, 0.261]))
    ).to(device)
    
    return train_aug, test_aug

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup Environment
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    # Enable CuDNN benchmark for T4 (fixed input size optimization)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Init WandB
    wandb.init(
        project=cfg.logger.project, 
        group="finetuning_strong", # Cambiato nome gruppo per distinguere
        name=f"ft_strong_{cfg.model.backbone}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 2. Data Preparation
    print("Loading data...")
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    
    train_aug, test_aug = get_strong_finetuning_transforms(device)

    # 3. Model Setup
    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)
    
    # --- LOAD PRE-TRAINED SELF-SUPERVISED WEIGHTS ---
    ckpt_path = root_dir / "checkpoints" / "self_supervised" / "last_model.pth"
    
    if ckpt_path.exists():
        print(f"Loading pre-trained weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("⚠️ WARNING: Pre-trained checkpoint not found! Training from scratch.")

    # --- SETUP CLASSIFIER HEAD ---
    if cfg.model.backbone == 'resnet18':
        feature_dim = 512
    else:
        feature_dim = 2048 
        
    num_classes = cfg.data.num_classes
    
    model.classifier = nn.Linear(feature_dim, num_classes).to(device)

    # --- LOAD LINEAR PROBE WEIGHTS ---
    probe_path = root_dir / "checkpoints" / "self_supervised" / "best_linear_probe.pth"
    
    if probe_path.exists():
        print("✓ Loading Linear Probe weights (Starting point optimized)...")
        probe_ckpt = torch.load(probe_path, map_location=device)
        model.classifier.load_state_dict(probe_ckpt['classifier_state_dict'])
    else:
        print("Linear probe checkpoint not found. Initializing classifier randomly.")

    # 4. FULL UNFREEZE
    for param in model.parameters():
        param.requires_grad = True
    
    print("✓ Model fully unfrozen. Ready for Fine-Tuning.")

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    # Mixed Precision Scaler for T4
    scaler = torch.amp.GradScaler('cuda')

    # 6. Training Loop
    ft_epochs = 30 # Puoi provare ad alzarlo a 50 se vedi che migliora ancora
    best_acc = 0.0

    print(f"\n--- Starting Fine-Tuning ({ft_epochs} epochs) ---\n")

    for epoch in range(ft_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{ft_epochs}")
        
        for imgs, labels in pbar:
            if isinstance(imgs, list): 
                imgs = imgs[0]
            
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Applicazione Augmentations CATTIVE su GPU
            with torch.no_grad():
                imgs = train_aug(imgs)
            
            # Mixed Precision Forward Pass
            with torch.amp.autocast('cuda'):
                features = model.backbone(imgs)
                features = torch.flatten(features, 1) 
                logits = model.classifier(features)
                loss = criterion(logits, labels)

            # Backward Pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            train_loss += loss.item()
            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            current_acc = train_correct / train_total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2%}'})

        # --- VALIDATION PHASE ---
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        
        with torch.no_grad():
            for imgs, labels in test_loader:
                if isinstance(imgs, list): imgs = imgs[0]
                
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Test Augmentation (Solo Normalize)
                imgs = test_aug(imgs) 
                
                with torch.amp.autocast('cuda'):
                    features = model.backbone(imgs)
                    features = torch.flatten(features, 1)
                    logits = model.classifier(features)
                    t_loss = criterion(logits, labels)
                
                test_loss += t_loss.item()
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += labels.size(0)
        
        test_acc = test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Epoch {epoch+1} Result | Train Acc: {train_correct/train_total:.2%} | Test Acc: {test_acc:.2%}")
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "ft/train_loss": train_loss / len(train_loader),
            "ft/train_acc": train_correct / train_total,
            "ft/test_loss": avg_test_loss,
            "ft/test_acc": test_acc
        })

        # Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = root_dir / "checkpoints" / "self_supervised" / "finetuned_best.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc
            }, save_path)
            print(f"  ✓ New best model saved! ({best_acc:.2%})")

    wandb.finish()

if __name__ == "__main__":
    main()