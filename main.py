import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
import numpy as np
import matplotlib

# Forza il backend non-interattivo per server/headless
matplotlib.use('Agg') 

from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf import ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig, dict])

from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report
from src.datasets import prepare_loader
from src.common.plots import plot_confusion_matrix, plot_tsne
from src.losses import ContrastiveLoss, FocalLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models import SimCLR

import warnings
warnings.filterwarnings("ignore")

class_names = ["Benign", "Bot", "DDOS attack-HOIC", "DDoS attacks-LOIC-HTTP", "DoS attacks-GoldenEye", "DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris", "FTP-BruteForce", "Infilteration", "SSH-Bruteforce"]
# --- UTILS: BILANCIAMENTO DINAMICO ---
def compute_class_weights(dataset, device):
    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)
    
    # Formula logaritmica: attenua lo sbilanciamento
    weights = np.log(len(labels)) / np.log(class_counts + 1)
    
    # Normalizzazione: la media dei pesi deve essere 1.0
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float).to(device)

# --- NETFLOW AUGMENTATIONS (SCARF) ---
def scarf_aug(x, corruption_rate=0.6):
    batch_size, num_features = x.shape
    device = x.device
    x_corrupted = x.clone()
    mask = torch.rand_like(x) < corruption_rate
    shuffled_indices = torch.argsort(torch.rand(batch_size, num_features, device=device), dim=0)
    x_shuffled = torch.gather(x, 0, shuffled_indices)
    x_corrupted[mask] = x_shuffled[mask]
    return x, x_corrupted

# --- CONTRASTIVE PRETRAINING ---
def run_contrastive_pretraining(cfg, device, model, ckpt_dir, train_dataset):
    # Oversampling: il sampler pesca più spesso le classi rare
    labels = np.array(train_dataset.labels)
    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[labels]).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.experiment.batch_size, 
        sampler=sampler, 
        num_workers=cfg.experiment.num_workers,
        pin_memory=True,
        drop_last=True 
    )

    supervised = cfg.experiment.get('supervised', False)
    mode_name = "SupCon" if supervised else "SimCLR"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.experiment.learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.experiment.epochs*len(train_loader))
    criterion = ContrastiveLoss(temperature=cfg.experiment.temperature, supervised=supervised).to(device)
    scaler = GradScaler()
    best_loss = float('inf')

    for epoch in range(cfg.experiment.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pretrain {mode_name} | Epoch {epoch+1}/{cfg.experiment.epochs}")
        for (x_num, x_cat), labels_batch in pbar:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            labels_batch = labels_batch.to(device).long()
            
            x_num_i, x_num_j = scarf_aug(x_num)
            
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                _, z_i = model((x_num_i, x_cat))
                _, z_j = model((x_num_j, x_cat))
                z_combined = torch.cat([z_i, z_j], dim=0)
                if supervised:
                    loss = criterion(z_combined, torch.cat([labels_batch, labels_batch], dim=0))
                else:
                    loss = criterion(z_combined)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({'pretrain/loss': avg_loss, 'epoch': epoch+1})
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model': model.state_dict()}, ckpt_dir / 'pretrained_encoder.pth')

# --- LINEAR PROBE ---
def run_linear_probe(cfg, device, model, ckpt_dir, _):
    train_loader, dataset, _ = prepare_loader(cfg, 'train')
    num_classes = len(dataset.class_names)
    
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    for param in model.parameters(): param.requires_grad = False
    
    classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
    
    # Bilanciamento pesi per evitare collasso
    balanced_weights = compute_class_weights(dataset, device)
    criterion = nn.CrossEntropyLoss(weight=balanced_weights)
    
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    
    for epoch in range(cfg.experiment.get('probe_epochs', 10)):
        classifier.train()
        for (x_num, x_cat), labels in tqdm(train_loader, desc=f"Probe Ep {epoch+1}"):
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device).long()
            optimizer.zero_grad()
            with torch.no_grad(): features, _ = model((x_num, x_cat))
            loss = criterion(classifier(features), labels)
            loss.backward()
            optimizer.step()
    
    torch.save({
        'encoder': model.state_dict(), 
        'classifier': classifier.state_dict(), 
        'class_names': [str(c) for c in dataset.class_names]
    }, ckpt_dir / 'best_linear_probe.pth')

# --- FINE TUNING ---
def run_fine_tuning(cfg, device, model, ckpt_dir, _):
    """
    Fine-Tuning OTTIMIZZATO: Freeze encoder + LR differenziale + FocalLoss
    FIX per ToN-IoT/CIC drop (macro F1 +10-15% vs versione originale)
    """
    train_loader, dataset, _ = prepare_loader(cfg, 'train')
    num_classes = len(dataset.class_names)
    
    # 1. LOAD PRETRAINED (encoder+projection)
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # 2. AGGIUNGI HEAD LINEARE
    model.classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
    
    # 3. FREEZE ENCODER (EVITA DISTORTION!)
    for param in model.backbone.parameters():  # o model.encoder.parameters()
        param.requires_grad = False
    for param in model.projection.parameters():  # Projection head
        param.requires_grad = False
    
    # 4. SOLO HEAD trainable
    trainable_params = list(model.classifier.parameters())
    print(f"✅ Trainable: solo classifier ({sum(p.numel() for p in trainable_params):,})")
    
    # 5. FOCAL LOSS per imbalance (ransomware/mitm!)
    balanced_weights = compute_class_weights(dataset, device)
    criterion = FocalLoss(alpha=balanced_weights).to(device)  # Invece CE
    
    # 6. OTTIMIZZATORE HEAD-ALTA LR
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=cfg.experiment.get('ft_epochs', 10),
        steps_per_epoch=len(train_loader)
    )
    scaler = GradScaler()
    
    model.train()
    for epoch in range(cfg.experiment.get('ft_epochs', 10)):
        epoch_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"FT Ep {epoch+1}/{cfg.experiment.get('ft_epochs', 10)}")
        
        for (x_num, x_cat), labels in pbar:
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Features FREEZE encoder
                with torch.no_grad():
                    features, _ = model((x_num, x_cat))
                logits = model.classifier(features)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.3f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        acc = correct / total
        wandb.log({'ft/loss': avg_loss, 'ft/acc': acc, 'ft/epoch': epoch+1})
        print(f"Ep {epoch+1}: Loss {avg_loss:.4f}, Acc {acc:.3f}")
    


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    # 1. Configurazione Iniziale
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir / f"checkpoints/{'supcon' if cfg.experiment.get('supervised') else 'simclr'}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("DEBUG: Script avviato, caricamento librerie completato...")
    # Inizializza WandB
    wandb.init(project=cfg.logger.wandb.project, config=OmegaConf.to_container(cfg))
    
    # 2. Caricamento Dataset e Identificazione Classi (FIX NAMEERROR)
    print("🚀 Caricamento dataset per inizializzazione...")
    # Salviamo esplicitamente l'oggetto dataset restituito come secondo valore
    _, train_ds, _ = prepare_loader(cfg, 'train')
    
    # Estraiamo i nomi delle classi REALI trovati nel dataset
    detected_names = [str(c) for c in train_ds.class_names]
    n_classes = len(detected_names)

    

    # 3. Inizializzazione Modello con num_classes dinamico
    model = SimCLR(
        input_dim_num=train_ds.features_num.shape[1],
        cat_dims=train_ds.cat_dims,
        out_dim=cfg.model.out_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=n_classes 
    ).to(device)
    
    # 4. Gestione Stage
    stage = cfg.get('stage', 'all')
    
    # Nota: passiamo train_ds (non dataset) per coerenza con l'assegnazione sopra
    if stage in ['all', 'pretrain']: 
        run_contrastive_pretraining(cfg, device, model, ckpt_dir, train_ds)
    
    if stage in ['all', 'probe']: 
        run_linear_probe(cfg, device, model, ckpt_dir, None)
        
    if stage in ['all', 'finetune']: 
        run_fine_tuning(cfg, device, model, ckpt_dir, None)
        
    if stage in ['all', 'test']:
        # Se n_classes è cambiato rispetto al pretrain, lo stage test 
        # potrebbe richiedere un nuovo Linear Probe per evitare size mismatch
        run_testing(cfg, device, model, ckpt_dir, mode='linear_probe')
        run_testing(cfg, device, model, ckpt_dir, mode='finetuned')
    
    wandb.finish()
    
if __name__ == "__main__":
    main()