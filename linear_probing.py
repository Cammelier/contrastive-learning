"""
Linear Probing per valutare le rappresentazioni apprese dal contrastive learning.
Freeze l'encoder e addestra solo un linear classifier.
"""

import hydra 
import torch 
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 

from src.datasets import prepare_loader
from src.models import SimCLR


@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup
    root_dir = Path(hydra.utils.get_original_cwd())
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    # 2. Carica checkpoint del modello pretrained
    checkpoint_path = cfg.get('checkpoint_path', None)
    if checkpoint_path is None:
        # Usa l'ultimo checkpoint disponibile
        ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
        checkpoints = list(ckpt_dir.glob("model_epoch_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # 3. Wandb per linear probing
    wandb.init(
        project=cfg.logger.project,
        group=f"{cfg.experiment.mode}_linear_probing",
        name=f"linear_probe_seed{cfg.seed}",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 4. Carica dataloaders (train e test con label)
    print("Loading data...")
    train_loader = prepare_loader(cfg, split='train')
    test_loader = prepare_loader(cfg, split='test')
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 5. Carica encoder pretrained e FREEZALO
    print("Loading pretrained encoder...")
    encoder = SimCLR(
        base_model=cfg.model.backbone, 
        out_dim=cfg.model.out_dim
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # FREEZE encoder - non verrà addestrato
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    print("✓ Encoder frozen (no gradient updates)")
    
    # 6. Linear classifier (solo questo verrà addestrato)
    feature_dim = 512  # ResNet18
    num_classes = cfg.data.num_classes
    
    linear_classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    print(f"Linear classifier: {feature_dim} -> {num_classes}")
    print(f"Trainable parameters: {sum(p.numel() for p in linear_classifier.parameters())}")
    
    # 7. Optimizer e loss (solo per classifier)
    optimizer = torch.optim.Adam(
        linear_classifier.parameters(),
        lr=cfg.get('linear_probe_lr', 0.001)
    )
    criterion = nn.CrossEntropyLoss()
    
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
            # Usa solo la prima view (non serve contrastive)
            x = imgs[0].to(device) if isinstance(imgs, list) else imgs.to(device)
            labels = labels.to(device)
            
            # Estrai features con encoder frozen
            with torch.no_grad():
               features = encoder(x, return_features=True)

            
            # Forward attraverso linear classifier
            logits = linear_classifier(features)
            loss = criterion(logits, labels)
            
            # Backward (solo classifier)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            current_acc = train_correct / train_total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })
        
        # Train epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # TEST (evaluation)
        linear_classifier.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
                x = imgs[0].to(device) if isinstance(imgs, list) else imgs.to(device)
                labels = labels.to(device)
                
                features = encoder(x, return_features=True)

                logits = linear_classifier(features)
                loss = criterion(logits, labels)
                
                test_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/accuracy": train_accuracy,
            "test/loss": avg_test_loss,
            "test/accuracy": test_accuracy,
        })
        
        # Print results
        print(f"Epoch [{epoch+1}/{linear_probe_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
        
        # Save best model
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_model_path = root_dir / "checkpoints" / cfg.experiment.mode / "best_linear_probe.pth"
            torch.save({
                'epoch': epoch + 1,
                'classifier_state_dict': linear_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_accuracy,
            }, best_model_path)
            print(f"  ✓ New best accuracy: {best_test_acc:.4f}")
    
    # Final evaluation con confusion matrix
    print(f"\n--- Final Evaluation ---")
    linear_classifier.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Eval"):
            x = imgs[0].to(device) if isinstance(imgs, list) else imgs.to(device)
            features = encoder(x, return_features=True)
            logits = linear_classifier(features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Per-class accuracy
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    print(f"\nPer-class Accuracy:")
    for i, acc in enumerate(per_class_acc):
        print(f"  Class {i}: {acc:.4f}")
    
    # Log confusion matrix to wandb
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Acc: {best_test_acc:.4f})')
    
    wandb.log({
        "final/best_test_accuracy": best_test_acc,
        "final/confusion_matrix": wandb.Image(fig),
        "final/mean_per_class_acc": np.mean(per_class_acc)
    })
    
    plt.close()
    wandb.finish()


if __name__ == "__main__":
    main() 