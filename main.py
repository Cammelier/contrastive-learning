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

class_names= ["backdoor", 
    "ddos", 
    "dos", 
    "injection", 
    "mitm", 
    "normal", 
    "password", 
    "scanning", 
    "vulnerability", 
    "xss"]

# --- UTILS: BILANCIAMENTO DINAMICO ---
def compute_class_weights(dataset, device):
    """
    Calcola i pesi per bilanciare le classi: peso = N_totale / (N_classi * N_campioni_classe)
    Garantisce che ogni classe abbia lo stesso impatto sulla loss.
    """
    labels = np.array(dataset.labels)
    num_classes = len(dataset.class_names)
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Evitiamo divisioni per zero per classi mancanti nello split
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    weights = torch.tensor(len(labels) / (num_classes * class_counts), dtype=torch.float)
    return weights.to(device)

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
    train_loader, dataset, _ = prepare_loader(cfg, 'train')
    num_classes = len(dataset.class_names)
    
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
    
    balanced_weights = compute_class_weights(dataset, device)

    criterion = nn.CrossEntropyLoss(weight=balanced_weights)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = GradScaler()
    
    for epoch in range(cfg.experiment.get('ft_epochs', 10)):
        model.train()
        for (x_num, x_cat), labels in tqdm(train_loader, desc=f"FT Ep {epoch+1}"):
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device).long()
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                h, _ = model((x_num, x_cat))
                loss = criterion(model.classifier(h), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    torch.save({
        'model': model.state_dict(), 
        'class_names': [str(c) for c in dataset.class_names]
    }, ckpt_dir / 'best_finetuned.pth')


# --- TESTING ---
def run_testing(cfg, device, model, ckpt_dir, mode='finetuned'):
    path = ckpt_dir / ('best_finetuned.pth' if mode=='finetuned' else 'best_linear_probe.pth')
    if not path.exists(): return
    checkpoint = torch.load(path, map_location=device)
    
    # Nomi delle classi salvati durante il training (10 classi)
    full_class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(10)])
    num_classes = len(full_class_names)

    model.load_state_dict(checkpoint['model'] if mode=='finetuned' else checkpoint['encoder'], strict=False)
    
    if mode == 'linear_probe':
        classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
        classifier.load_state_dict(checkpoint['classifier'])
    else:
        classifier = model.classifier

    test_loader, _, _ = prepare_loader(cfg, 'test')
    model.eval(); classifier.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for (x_num, x_cat), labels in tqdm(test_loader, desc=f"Testing {mode}"):
            h, _ = model((x_num.to(device), x_cat.to(device)))
            all_preds.extend(classifier(h).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # --- FIX PER IL REPORT ---
    # Identifichiamo quali indici di classe sono effettivamente presenti nei dati di test
    present_indices = np.unique(np.concatenate([all_labels, all_preds]))
    # Filtriamo i nomi delle classi basandoci solo sugli indici presenti
    present_names = [full_class_names[i] for i in present_indices]
    
    print(f"\n📊 REPORT {mode.upper()}:\n")
    print(classification_report(
        all_labels, 
        all_preds, 
        labels=present_indices,    # Dice a sklearn quali numeri cercare
        target_names=present_names, # Associa i nomi corretti a quei numeri
        digits=4
    ))
    # -------------------------
    
    plot_confusion_matrix(all_labels, all_preds, full_class_names, mode)
    plot_tsne(model, test_loader, device, 0, mode, full_class_names)

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir / f"checkpoints/{'supcon' if cfg.experiment.get('supervised') else 'simclr'}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project=cfg.logger.wandb.project, config=OmegaConf.to_container(cfg))
    
    # Caricamento train per inizializzazione
    _, train_dataset, _ = prepare_loader(cfg, 'train')
    num_classes = len(train_dataset.class_names)
    
    model = SimCLR(
        input_dim_num=train_dataset.features_num.shape[1],
        cat_dims=train_dataset.cat_dims,
        out_dim=cfg.model.out_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    stage = cfg.get('stage', 'all')
    if stage in ['all', 'pretrain']: run_contrastive_pretraining(cfg, device, model, ckpt_dir, train_dataset)
    if stage in ['all', 'probe']: run_linear_probe(cfg, device, model, ckpt_dir, None)
    if stage in ['all', 'finetune']: run_fine_tuning(cfg, device, model, ckpt_dir, None)
    if stage in ['all', 'test']:
        run_testing(cfg, device, model, ckpt_dir, mode='linear_probe')
        run_testing(cfg, device, model, ckpt_dir, mode='finetuned')
    
    wandb.finish()

if __name__ == "__main__":
    main()