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
from sklearn.metrics import f1_score, classification_report


import warnings
warnings.filterwarnings("ignore")

# --- NETFLOW AUGMENTATIONS ---
def netflow_aug(x, noise_strength=0.02, drop_prob=0.1):
    """Generate 2 augmented views for NetFlow vectors [B, features]"""
    noise = noise_strength * torch.randn_like(x)
    
    # View 1: 10% feature dropout + noise
    mask1 = torch.rand_like(x) > drop_prob
    x_i = x * mask1 + noise * (~mask1)
    
    # View 2: 15% different dropout + noise
    mask2 = torch.rand_like(x) > (drop_prob + 0.05)
    x_j = x * mask2 + noise * (~mask2)
    
    return x_i, x_j


# --- CONTRASTIVE PRETRAINING (Self-Supervised or SupCon) ---
def run_contrastive_pretraining(cfg: DictConfig, device, model, ckpt_dir: Path):
    """
    Pure contrastive pretraining WITHOUT classifier.
    
    - If supervised=False: SimCLR (labels ignored)
    - If supervised=True: SupCon (labels used for positive pairs)
    """
    # 1. DATA
    train_loader, _ = prepare_loader(cfg, 'train')
    
    supervised = cfg.experiment.get('supervised', False)
    mode_name = "SupCon" if supervised else "SimCLR"
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting {mode_name} Contrastive Pretraining")
    print(f"{'='*60}\n")
    
    # 2. OPTIMIZER & SCHEDULER
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.experiment.learning_rate,
        weight_decay=cfg.experiment.get('weight_decay', 1e-4)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=cfg.experiment.learning_rate, 
        epochs=cfg.experiment.epochs, 
        steps_per_epoch=len(train_loader)
    )
    
    # 3. CONTRASTIVE LOSS
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=supervised  
    ).to(device)
    
    scaler = GradScaler('cuda')
    
    # 4. TRAINING LOOP
    for epoch in range(cfg.experiment.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        for batch_idx, (x, labels) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Generate augmented views
                x_i, x_j = netflow_aug(x)
                
                # Forward pass through both views
                h_combined, z_combined = model(torch.cat([x_i, x_j], dim=0))
                
                # Contrastive Loss
                if supervised:
                    # SupCon: use labels to define positive pairs
                    labels_combined = torch.cat([labels, labels], dim=0)
                    loss = criterion(z_combined, labels_combined)
                else:
                    # SimCLR: augmented views are positives
                    loss = criterion(z_combined)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
        
        # Logging
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({
            'epoch': epoch + 1,
            'train/contrastive_loss': avg_loss,
            'train/lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
    
    # 5. SAVE ENCODER ONLY (no classifier)
    torch.save({
        'model': model.state_dict(),
        'config': OmegaConf.to_container(cfg),
        'mode': mode_name
    }, ckpt_dir / 'pretrained_encoder.pth')
    
    print(f"\n✅ {mode_name} Pretraining Complete! Encoder saved.\n")


# --- LINEAR PROBING EVALUATION ---


def run_linear_probe(cfg: DictConfig, device, model, ckpt_dir: Path):
    """
    Evaluate pretrained encoder with frozen weights + linear classifier.
    This is the TRUE test of self-supervised quality.
    """
    print(f"\n{'='*60}")
    print(f"🔬 Linear Probing Evaluation (Encoder Frozen)")
    print(f"{'='*60}\n")
    
    # 1. DATA
    train_loader, _ = prepare_loader(cfg, 'train')
    val_loader, class_names = prepare_loader(cfg, 'val')
    
    num_classes = len(class_names)
    
    # 2. LOAD PRETRAINED ENCODER
    # Assicurati che il nome del file coincida con quello salvato nel pretraining
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"✅ Loaded pretrained encoder from: {checkpoint.get('mode', 'Unknown')}")
    
    # 3. FREEZE ENCODER
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # Important: freeze BatchNorm statistics
    
    # 4. NEW LINEAR CLASSIFIER
    classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
    
    # 5. OPTIMIZER (only for classifier)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=cfg.experiment.get('probe_lr', 0.01),
        momentum=0.9,
        weight_decay=0
    )
    
    probe_epochs = cfg.experiment.get('probe_epochs', 20)
    criterion = nn.CrossEntropyLoss()
    
    # 6. TRAINING LOOP (classifier only)
    best_val_f1_macro = 0.0
    
    for epoch in range(probe_epochs):
        classifier.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{probe_epochs}")
        
        for x, labels in pbar:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Extract features (no augmentation for probing)
            with torch.no_grad():
                features, _ = model(x)
            
            # Classify
            logits = classifier(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Accuracy
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix(acc=f'{train_correct/train_total:.2%}')
        
        # 7. VALIDATION
        classifier.eval()
        val_correct, val_total = 0, 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                
                features, _ = model(x)
                logits = classifier(features)
                
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Raccogli per F1-Score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcolo metriche avanzate
        val_acc = val_correct / val_total if val_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        # Log su WandB
        wandb.log({
            'probe_epoch': epoch + 1,
            'probe/train_acc': train_acc,
            'probe/val_acc': val_acc,
            'probe/val_f1_macro': f1_macro,
            'probe/val_f1_weighted': f1_weighted
        })
        
        # Salvataggio basato su F1 Macro (più affidabile dell'accuracy per sbilanciamento)
        if f1_macro > best_val_f1_macro:
            best_val_f1_macro = f1_macro
            torch.save({
                'encoder': model.state_dict(),
                'classifier': classifier.state_dict(),
                'class_names': class_names,
                'val_f1_macro': f1_macro,
                'val_acc': val_acc
            }, ckpt_dir / 'best_linear_probe.pth')
        
        # Print periodico e report finale
        if (epoch + 1) % 5 == 0 or epoch == probe_epochs - 1:
            print(f"\nProbe Epoch {epoch+1}: Val Acc = {val_acc:.2%}, F1 Macro = {f1_macro:.4f}")
            if epoch == probe_epochs - 1:
                print("\n📊 FINAL CLASSIFICATION REPORT:")
                print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print(f"\n✅ Linear Probe Complete! Best Val F1 Macro: {best_val_f1_macro:.4f}\n")
    return best_val_f1_macro





# --- FINE-TUNING ---
def run_fine_tuning(cfg: DictConfig, device, model, ckpt_dir: Path):
    """
    Fine-tune the entire model (encoder + classifier) end-to-end.
    Aggiornato con monitoraggio F1-Score Macro.
    """
    print(f"\n{'='*60}")
    print(f"🎯 Fine-Tuning (End-to-End Training)")
    print(f"{'='*60}\n")
    
    # 1. DATA
    train_loader, _ = prepare_loader(cfg, 'train')
    val_loader, class_names = prepare_loader(cfg, 'val')
    num_classes = len(class_names)
    
    # 2. LOAD PRETRAINED ENCODER
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # 3. ADD CLASSIFIER HEAD
    model.classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
    
    # 4. DIFFERENTIAL LEARNING RATES
    # Usiamo un LR molto basso per il backbone e più alto per la testa
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.experiment.get('ft_lr_backbone', 1e-5)},
        {'params': model.projection.parameters(), 'lr': cfg.experiment.get('ft_lr_backbone', 1e-5)},
        {'params': model.classifier.parameters(), 'lr': cfg.experiment.get('ft_lr_head', 1e-4)}
    ], weight_decay=cfg.experiment.get('weight_decay', 1e-4))
    
    ft_epochs = cfg.experiment.get('ft_epochs', 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() # Per Mixed Precision
    
    best_val_f1_macro = 0.0
    
    # 5. TRAINING LOOP
    for epoch in range(ft_epochs):
        model.train()
        train_correct, train_total = 0, 0
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{ft_epochs}")
        
        for x, labels in pbar:
            x, labels = x.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            # Utilizzo di autocast per velocizzare il training
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Augmentazione leggera per il fine-tuning
                x_aug, _ = netflow_aug(x, noise_strength=0.01, drop_prob=0.05)
                features, _ = model(x_aug)
                logits = model.classifier(features)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(acc=f'{train_correct/train_total:.2%}')
        
        scheduler.step()
        
        # 6. VALIDATION
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                features, _ = model(x)
                logits = model.classifier(features)
                
                preds = logits.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calcolo Metriche
        val_acc = val_correct / val_total if val_total > 0 else 0
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        wandb.log({
            'ft_epoch': epoch + 1,
            'ft/train_acc': train_correct / train_total,
            'ft/val_acc': val_acc,
            'ft/val_f1_macro': f1_macro,
            'ft/val_f1_weighted': f1_weighted
        })
        
        # Salvataggio basato su F1 Macro
        if f1_macro > best_val_f1_macro:
            best_val_f1_macro = f1_macro
            torch.save({
                'model': model.state_dict(),
                'class_names': class_names,
                'val_f1_macro': f1_macro,
                'val_acc': val_acc
            }, ckpt_dir / 'best_finetuned.pth')
        
        if (epoch + 1) % 5 == 0 or epoch == ft_epochs - 1:
            print(f"FT Epoch {epoch+1}: Val Acc = {val_acc:.2%}, F1 Macro = {f1_macro:.4f}")
            if epoch == ft_epochs - 1:
                print("\n📝 CLASSIFICATION REPORT (Fine-Tuning):")
                print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    
    print(f"\n✅ Fine-Tuning Complete! Best Val F1 Macro: {best_val_f1_macro:.4f}\n")
    return best_val_f1_macro



# --- TESTING ---
def run_testing(cfg: DictConfig, device, model, ckpt_dir: Path, mode='linear_probe'):
    """
    Final evaluation on test set.
    
    Args:
        mode: 'linear_probe' or 'finetuned'
    """
    print(f"\n{'='*60}")
    print(f"🧪 Testing ({mode.upper()})")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    if mode == 'linear_probe':
        checkpoint_path = ckpt_dir / 'best_linear_probe.pth'
    else:
        checkpoint_path = ckpt_dir / 'best_finetuned.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 0.0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model
    if mode == 'linear_probe':
        model.load_state_dict(checkpoint['encoder'], strict=False)
        classifier = nn.Linear(cfg.model.hidden_dim, len(checkpoint['class_names'])).to(device)
        classifier.load_state_dict(checkpoint['classifier'])
    else:
        model.load_state_dict(checkpoint['model'])
        classifier = model.classifier
    
    class_names = checkpoint['class_names']
    
    # Test data
    test_loader, _ = prepare_loader(cfg, 'test')
    
    model.eval()
    classifier.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc='Testing'):
            x, labels = x.to(device), labels.to(device)
            
            features, _ = model(x)
            logits = classifier(features)
            preds = logits.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100 * np.mean(all_preds == all_labels)
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS ({mode.upper()})")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Log to wandb
    wandb.log({
        f"test/{mode}_acc": acc,
        f"test/{mode}_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, 
            y_true=all_labels, 
            preds=all_preds, 
            class_names=class_names
        )
    })
    
    return acc


# --- MAIN ENTRY POINT ---
setup_logger() 

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # Determine mode
    supervised = cfg.experiment.get('supervised', False)
    mode_name = "supcon" if supervised else "simclr"
    
    ckpt_dir = root_dir / f"checkpoints/{mode_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)
    
    # Initialize wandb
    wandb.init(
        project=cfg.logger.wandb.project,
        name=f"{mode_name}_{cfg.data.file_name}",
        config=OmegaConf.to_container(cfg),
        tags=[mode_name, cfg.data.file_name]
    )
    
    # Initialize model
    model = SimCLR(
        input_dim=cfg.data.input_dim,  
        hidden_dim=cfg.model.hidden_dim,
        out_dim=cfg.model.out_dim
    ).to(device)
    
    # Pipeline stages
    stage = cfg.get('stage', 'all')
    
    if stage in ['all', 'pretrain']:
        run_contrastive_pretraining(cfg, device, model, ckpt_dir)
    
    if stage in ['all', 'probe']:
        probe_acc = run_linear_probe(cfg, device, model, ckpt_dir)
    
    if stage in ['all', 'finetune']:
        ft_acc = run_fine_tuning(cfg, device, model, ckpt_dir)
    
    if stage in ['all', 'test_probe']:
        test_acc_probe = run_testing(cfg, device, model, ckpt_dir, mode='linear_probe')
    
    if stage in ['all', 'test_ft']:
        test_acc_ft = run_testing(cfg, device, model, ckpt_dir, mode='finetuned')
    
    wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"✅ ALL STAGES COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()