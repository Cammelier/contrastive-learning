import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from omegaconf import DictConfig, ListConfig

torch.serialization.add_safe_globals([DictConfig, ListConfig, dict])

from pathlib import Path
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score
from src.datasets import prepare_loader
from src.common.plot import plot_enhanced_confusion_matrix 
from src.losses import ContrastiveLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast 
from src.models import SimCLR


import warnings
warnings.filterwarnings("ignore")

# --- NETFLOW AUGMENTATIONS ---
def netflow_aug(x, noise_strength=0.02, drop_prob=0.1, corruption_rate=0.6):
    batch_size, num_features = x.shape
    device = x.device
    
    x_corrupted = x.clone()
    
    mask = torch.rand_like(x) < corruption_rate
    
    shuffled_indices = torch.argsort(torch.rand(batch_size, num_features, device=device), dim=0)
    
    x_shuffled = torch.gather(x, 0, shuffled_indices)
    
    x_corrupted[mask] = x_shuffled[mask]
    
    return x, x_corrupted

# --- CONTRASTIVE PRETRAINING ---
def run_contrastive_pretraining(cfg, device, model, ckpt_dir):
    # 1. BILANCIAMENTO SOTA: Creazione del Sampler per gestire lo sbilanciamento estremo
    train_loader, train_dataset, loss_weights = prepare_loader(cfg, 'train')

    labels = np.array(train_dataset.labels) # Assumi che il dataset abbia un attributo .labels
    class_sample_count = np.unique(labels, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = torch.from_numpy(weight[labels]).double()
    
    # Il sampler forza ogni batch ad avere una rappresentazione equa delle 9 classi
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.experiment.batch_size, 
        sampler=sampler, 
        num_workers=cfg.experiment.num_workers,
        pin_memory=True,
        drop_last=True # Cruciale per BatchNorm in SimCLR
    )

    supervised = cfg.experiment.get('supervised', False)
    mode_name = "SupCon" if supervised else "SimCLR"
    
    # Regolazione automatica della temperatura se bloccata
    temp = cfg.experiment.temperature
    if supervised and temp < 0.1:
        temp = 0.15 
    
    print(f"\n🚀 Starting {mode_name} Pretraining | Temp: {temp} | Balanced Sampling: ON")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.experiment.learning_rate, 
        weight_decay=0.05 if not supervised else 0.01
    )
    
    
   # total_steps = cfg.experiment.epochs * len(train_loader)

    # Scheduler OneCycle 
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #    optimizer,
    #    max_lr=cfg.experiment.learning_rate,
    #    total_steps=total_steps,
    #    pct_start=0.1,  
    #    div_factor=25,  
    #    final_div_factor=1000, 
    #    anneal_strategy='cos'
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.experiment.epochs*len(train_loader))

    
    
    criterion = ContrastiveLoss(temperature=temp, supervised=supervised).to(device)
    scaler = GradScaler()
    best_loss = float('inf')

    for epoch in range(cfg.experiment.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pretrain {mode_name} | Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        for x, labels in pbar:
            x, labels = x.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            
            # Utilizzo di Mixed Precision per velocità e stabilità
            with autocast(dtype=torch.bfloat16):
                # Augmentation: x_i e x_j sono due viste dello stesso batch bilanciato
                x_i, x_j = netflow_aug(x, corruption_rate=0.6) 
                
                # Forward unico su batch concatenato
                _, z_combined = model(torch.cat([x_i, x_j], dim=0))
                
                if supervised:
                    # In SupCon, labels_combined guida il raggruppamento dei cluster
                    labels_combined = torch.cat([labels, labels], dim=0)
                    loss = criterion(z_combined, labels_combined)
                else:
                    # In SimCLR, la loss è puramente strutturale
                    loss = criterion(z_combined)
            
            # Verifica stabilità numerica
            if torch.isnan(loss):
                print("⚠️ Warning: NaN loss detected, skipping step")
                continue

            scaler.scale(loss).backward()
            
            # Gradient Clipping per evitare esplosioni in SupCon
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.6f}', lr=f'{scheduler.get_last_lr()[0]:.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({'pretrain/loss': avg_loss, 'pretrain/epoch': epoch+1, 'pretrain/lr': scheduler.get_last_lr()[0]})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Salvataggio modello con metadati
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
                'mode': mode_name,
                'config': cfg
            }
            torch.save(state, ckpt_dir / 'pretrained_encoder.pth')

    print(f"✨ Pretraining {mode_name} completato. Best Loss: {best_loss:.4f}")
def run_linear_probe(cfg: DictConfig, device, model, ckpt_dir: Path, loss_weights: torch.Tensor):
    print(f"\n🔬 Linear Probing (Encoder Frozen)")
    # Usa il loader bilanciato che abbiamo scritto prima!
    train_loader, dataset, _ = prepare_loader(cfg, 'train')
    val_loader, _, _ = prepare_loader(cfg, 'val')
    class_names = dataset.class_names
    
    # 1. Caricamento pesi con strict=False per ignorare il classificatore mancante
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False) # <--- AGGIUNTO strict=False
    
    # 2. Congelamento TOTALE dell'encoder (Backbone + Projection)
    for param in model.parameters(): 
        param.requires_grad = False
    model.eval()
    
    # 3. Definiamo il classificatore (usiamo quello già nel modello se esiste, o uno nuovo)
    # Assicurati che hidden_dim sia 512 come nel tuo modello
    classifier = nn.Linear(cfg.model.hidden_dim, len(class_names)).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device),
                                label_smoothing=0.2)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=cfg.experiment.get('probe_lr', 1e-3))
    
    best_f1 = 0.0
    for epoch in range(cfg.experiment.get('probe_epochs', 20)):
        classifier.train()
        for x, labels in tqdm(train_loader, desc=f"Probe Epoch {epoch+1}/{cfg.experiment.get('probe_epochs', 20)}"):
            x, labels = x.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            
            # Estrazione feature senza calcolare gradienti (risparmio VRAM)
            with torch.no_grad():
                # features è l'output 'h' del tuo modello
                features, _ = model(x) 
            
            logits = classifier(features)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
        # --- VALIDAZIONE ---
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                h, _ = model(x)
                all_preds.extend(classifier(h).argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Uso di F1 Macro per valutare correttamente le classi rare (Recall 15%)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        from sklearn.metrics import balanced_accuracy_score
        b_acc = balanced_accuracy_score(all_labels, all_preds)
        
        print(f"📊 Probe Epoch {epoch+1} | F1 Macro: {f1:.4f} | Balanced Acc: {b_acc:.2%}")
        
        if f1 > best_f1:
            best_f1 = f1
            # Salvataggio checkpoint completo per il fine-tuning successivo
            torch.save({
                'encoder': model.state_dict(), 
                'classifier': classifier.state_dict(), 
                'class_names': class_names
            }, ckpt_dir / 'best_linear_probe.pth')
            
    return best_f1


# --- FINE-TUNING ---
def run_fine_tuning(cfg, device, model, ckpt_dir, loss_weights: torch.Tensor):
    print(f"\n🎯 Starting End-to-End Fine-Tuning | Target: SOTA Recall")
    
    # 1. Caricamento pesi dal Pre-training (SupCon o SimCLR)
    checkpoint = torch.load(ckpt_dir / 'pretrained_encoder.pth', map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    for param in model.parameters():
        param.requires_grad = True

    model.train()
    # 2. Re-inizializzazione Head (Classificatore lineare sulle feature 'h')
    train_loader, dataset, _ = prepare_loader(cfg, 'train')
    val_loader, _, _ = prepare_loader(cfg, 'val')
    class_names = dataset.class_names
    
    # 3. Loss Bilanciata con Label Smoothing (Fix per campioni rumorosi)
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device), 
        label_smoothing=cfg.experiment.get('label_smoothing', 0.1)
    )
    
    # 4. Optimizer con Learning Rate differenziato (Discriminative LR)
    # Proteggiamo il backbone (LR basso) e alleniamo la testa (LR alto)
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.experiment.get('ft_lr_backbone', 1e-6)},
        {'params': model.classifier.parameters(), 'lr': cfg.experiment.get('ft_lr_head', 1e-3)}
    ], weight_decay=0.01)

    best_f1 = 0.0
    scaler = GradScaler()
    epochs = cfg.experiment.get('ft_epochs', 20)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{epochs}")
        
        for x, labels in pbar:
            x, labels = x.to(device), labels.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(dtype=torch.bfloat16):
                # Otteniamo le feature h (non proiettate)
                h, _ = model(x) 
                logits = model.classifier(h)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            # Gradient Clipping per stabilità nelle classi rare
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- VALIDAZIONE ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                h, _ = model(x)
                logits = model.classifier(h)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metriche SOTA (Macro F1 pesa equamente tutte le classi)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        print(f"📈 FT Epoch {epoch+1} | Macro F1: {f1_macro:.4f}")
        
        wandb.log({'finetune/loss': train_loss/len(train_loader), 'finetune/f1_macro': f1_macro})

        if f1_macro > best_f1:
            best_f1 = f1_macro
            print(f"🔥 New Best F1 Macro! Saving model...")
            # Stampa il report dettagliato per vedere la Recall delle classi critiche
            print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
            
            torch.save({
                'model': model.state_dict(),
                'f1_macro': f1_macro,
                'class_names': class_names
            }, ckpt_dir / 'best_finetuned.pth')

    return best_f1

# --- TESTING ---
def run_testing(cfg, device, model, ckpt_dir: Path, mode='linear_probe'):
    checkpoint_path = ckpt_dir / ('best_linear_probe.pth' if mode == 'linear_probe' else 'best_finetuned.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint['class_names']
    
    if mode == 'linear_probe':
        model.load_state_dict(checkpoint['encoder'], strict=False)
        classifier = nn.Linear(cfg.model.hidden_dim, len(class_names)).to(device)
        classifier.load_state_dict(checkpoint['classifier'])
    else:
        model.load_state_dict(checkpoint['model'])
        classifier = model.classifier

    test_loader, _ , _ = prepare_loader(cfg, 'test')
    model.eval(); classifier.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc=f"Testing {mode}"):
            f, _ = model(x.to(device))
            all_preds.extend(classifier(f).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    plot_enhanced_confusion_matrix(all_labels, all_preds, class_names, mode)
    print(f"\n📊 REPORT {mode.upper()}:\n", classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    wandb.log({f"test_{mode}_f1": f1_score(all_labels, all_preds, average='macro')})

# --- MAIN ---
@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    supervised = cfg.experiment.get('supervised', False)
    mode_name = "supcon" if supervised else "simclr"
    ckpt_dir = root_dir / f"checkpoints/{mode_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project=cfg.logger.wandb.project, name=f"{mode_name}_{cfg.data.file_name}", config=OmegaConf.to_container(cfg))
    
     # Assicurati che SimCLR accetti questi parametri
    model = SimCLR(input_dim=cfg.data.input_dim, hidden_dim=cfg.model.hidden_dim, out_dim=cfg.model.out_dim).to(device)
    
    # 1. Carica pesi loss una volta sola
    _, _, loss_weights = prepare_loader(cfg, 'train')
    
    stage = cfg.get('stage', 'all')
    if stage in ['all', 'pretrain']: run_contrastive_pretraining(cfg, device, model, ckpt_dir)
         # PULIZIA MANUALE PER 2060
    del model # Eliminiamo l'istanza temporanea
    torch.cuda.empty_cache() # Svuotiamo la VRAM
    import gc
    gc.collect()
    
    # Ricarichiamo il modello fresco per il Probing
    model = SimCLR(input_dim=40, num_classes=9).to(device) 
    if stage in ['all', 'probe']: run_linear_probe(cfg, device, model, ckpt_dir, loss_weights)
    if stage in ['all', 'finetune']: run_fine_tuning(cfg, device, model, ckpt_dir, loss_weights)
    if stage in ['all', 'test']:
        run_testing(cfg, device, model, ckpt_dir, mode='linear_probe')
        run_testing(cfg, device, model, ckpt_dir, mode='finetuned')
    
    wandb.finish()

if __name__ == "__main__":
    main()
