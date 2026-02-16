import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf import ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig, dict])

from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score
from src.datasets import prepare_loader
from src.common.plots import plot_confusion_matrix, plot_tsne
from src.losses import ContrastiveLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models import SimCLR


import warnings
warnings.filterwarnings("ignore")

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
    # NOTA: Riceviamo train_dataset come argomento per evitare di ricaricarlo (risparmio tempo)
    
    # 1. BILANCIAMENTO SOTA: Creazione del Sampler
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
    
    temp = cfg.experiment.temperature
    if supervised and temp < 0.1: temp = 0.15 
    
    print(f"\n🚀 Starting {mode_name} Pretraining | Temp: {temp} | Balanced Sampling: ON")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.experiment.learning_rate, 
        weight_decay=0.05 if not supervised else 0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.experiment.epochs*len(train_loader))
    
    criterion = ContrastiveLoss(temperature=temp, supervised=supervised).to(device)
    scaler = GradScaler()
    best_loss = float('inf')

    for epoch in range(cfg.experiment.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pretrain {mode_name} | Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        # SimCLR ora si aspetta una tupla (x_num, x_cat) dal dataset
        for (x_num, x_cat), labels in pbar:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            labels = labels.to(device).long()
            
            # Nota: L'augmentation SCARF si applica principalmente ai numerici per ora
            # O se vuoi applicarla a tutto, dovresti concatenare prima. 
            # Per semplicità qui applichiamo SCARF solo alla parte numerica
            x_num_i, x_num_j = scarf_aug(x_num, corruption_rate=0.6)
            
            # Passiamo le tuple al modello: (num, cat)
            # Dobbiamo duplicare x_cat perché l'augmentation non lo tocca in questa versione base
            input_i = (x_num_i, x_cat)
            input_j = (x_num_j, x_cat)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Il modello gestisce internamente embedding e concatenazione
                _, z_i = model(input_i)
                _, z_j = model(input_j)
                z_combined = torch.cat([z_i, z_j], dim=0)
                
                if supervised:
                    labels_combined = torch.cat([labels, labels], dim=0)
                    loss = criterion(z_combined, labels_combined)
                else:
                    loss = criterion(z_combined)
            
            if torch.isnan(loss):
                print("⚠️ Warning: NaN loss detected")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({'pretrain/loss': avg_loss, 'pretrain/epoch': epoch+1})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'config': cfg
            }
            torch.save(state, ckpt_dir / 'pretrained_encoder.pth')

    print(f"✨ Pretraining {mode_name} completato. Best Loss: {best_loss:.4f}")

def run_linear_probe(cfg, device, model, ckpt_dir, loss_weights):
    print(f"\n🔬 Linear Probing")
    train_loader, dataset, _ = prepare_loader(cfg, 'train')
    val_loader, _, _ = prepare_loader(cfg, 'val')
    
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    for param in model.parameters(): param.requires_grad = False
    model.eval()
    
    classifier = nn.Linear(cfg.model.hidden_dim, len(dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3)
    
    for epoch in range(10): # Probe rapido
        classifier.train()
        for (x_num, x_cat), labels in tqdm(train_loader, desc=f"Probe Epoch {epoch+1}"):
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features, _ = model((x_num, x_cat))
            logits = classifier(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
    
    torch.save({'encoder': model.state_dict(), 'classifier': classifier.state_dict(), 'class_names': dataset.class_names}, ckpt_dir / 'best_linear_probe.pth')

def run_fine_tuning(cfg, device, model, ckpt_dir, loss_weights):
    print(f"\n🎯 Fine-Tuning End-to-End")
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    for param in model.parameters(): param.requires_grad = True
    
    train_loader, _, _ = prepare_loader(cfg, 'train')
    val_loader, dataset, _ = prepare_loader(cfg, 'val') # Serve dataset per class names
    model.classifier = nn.Linear(cfg.model.hidden_dim, len(dataset.class_names)).to(device)

    # Optimizer differenziato
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-6},
        {'params': model.embeddings.parameters(), 'lr': 1e-5}, # Embeddings lenti
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))
    scaler = GradScaler()
    
    for epoch in range(cfg.experiment.get('ft_epochs', 20)):
        model.train()
        for (x_num, x_cat), labels in tqdm(train_loader, desc=f"FT Epoch {epoch+1}"):
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                # Forward completo (SimCLR gestisce l'input tupla)
                h, _ = model((x_num, x_cat))
                logits = model.classifier(h)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation semplice
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for (x_num, x_cat), labels in val_loader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                h, _ = model((x_num, x_cat))
                logits = model.classifier(h)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"📈 FT Epoch {epoch+1} | Macro F1: {f1:.4f}")
        wandb.log({'finetune/f1': f1})
        
        torch.save({'model': model.state_dict(), 'class_names': dataset.class_names}, ckpt_dir / 'best_finetuned.pth')

def run_testing(cfg, device, model, ckpt_dir, mode='finetuned'):
    path = ckpt_dir / ('best_finetuned.pth' if mode=='finetuned' else 'best_linear_probe.pth')
    if not path.exists(): return
    checkpoint = torch.load(path, map_location=device)
    
    # Se probe, carica encoder e crea classifier a parte
    if mode == 'linear_probe':
        model.load_state_dict(checkpoint['encoder'], strict=False)
        classifier = nn.Linear(cfg.model.hidden_dim, len(checkpoint['class_names'])).to(device)
        classifier.load_state_dict(checkpoint['classifier'])
    else:
        model.load_state_dict(checkpoint['model'], strict=False)
        # Assicurati che il modello abbia il classificatore
        if not hasattr(model, 'classifier'):
             model.classifier = nn.Linear(cfg.model.hidden_dim, len(checkpoint['class_names'])).to(device)
        classifier = model.classifier

    test_loader, dataset, _ = prepare_loader(cfg, 'test')
    model.eval(); classifier.eval()
    
    preds, labels_list = [], []
    with torch.no_grad():
        for (x_num, x_cat), labels in tqdm(test_loader, desc=f"Testing {mode}"):
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            h, _ = model((x_num, x_cat))
            preds.extend(classifier(h).argmax(1).cpu().numpy())
            labels_list.extend(labels.numpy())
            
    print(f"\n📊 REPORT {mode.upper()}:\n", classification_report(labels_list, preds, target_names=dataset.class_names, digits=4))


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    supervised = cfg.experiment.get('supervised', False)
    mode_name = "supcon" if supervised else "simclr"
    ckpt_dir = root_dir / f"checkpoints/{mode_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project=cfg.logger.wandb.project, name=f"{mode_name}_{cfg.data.file_name}", config=OmegaConf.to_container(cfg))
    
    # 1. CARICAMENTO DATI (Prima del modello per avere le dimensioni!)
    print("📊 Loading Data for Model Init...")
    train_loader, train_dataset, loss_weights = prepare_loader(cfg, 'train')
    
    # 2. OTTENIAMO DIMENSIONI CORRETTE
    input_dim_num = train_dataset.features_num.shape[1]
    cat_dims = train_dataset.cat_dims
    num_classes = len(train_dataset.class_names)
    
    print(f"✅ Dimensions Detected: Num={input_dim_num}, Cat={cat_dims}")
    
    # 3. INIZIALIZZAZIONE MODELLO CORRETTA
    model = SimCLR(
        input_dim_num=input_dim_num,
        cat_dims=cat_dims,
        out_dim=cfg.model.out_dim,
        hidden_dim=cfg.model.hidden_dim
    ).to(device)
    
    stage = cfg.get('stage', 'all')
    if stage in ['all', 'pretrain']: 
        # Passiamo train_dataset per non ricaricarlo!
        run_contrastive_pretraining(cfg, device, model, ckpt_dir, train_dataset)
    
    # PULIZIA
    del model
    torch.cuda.empty_cache()
    
    # RICARICA FRESCA (Con dimensioni corrette)
    model = SimCLR(
        input_dim_num=input_dim_num,
        cat_dims=cat_dims,
        out_dim=cfg.model.out_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    if stage in ['all', 'probe']: run_linear_probe(cfg, device, model, ckpt_dir, loss_weights)
    if stage in ['all', 'finetune']: run_fine_tuning(cfg, device, model, ckpt_dir, loss_weights)
    if stage in ['all', 'test']:
        run_testing(cfg, device, model, ckpt_dir, mode='linear_probe')
        run_testing(cfg, device, model, ckpt_dir, mode='finetuned')
    
    wandb.finish()

if __name__ == "__main__":
    main()