import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.io import load_df

class NetFlowDataset(Dataset):
    def __init__(self, cfg, split='train'):
        base_path = Path(cfg.path.processed_data)
        file_path = base_path / f"{cfg.data.file_name}_{split}.{cfg.data.extension}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        df = load_df(str(file_path))
        
        num_cols = list(cfg.data.num_cols)
        cat_cols = list(cfg.data.cat_cols)
        
        label_col = f"multi_{cfg.data.label_col}" 
        if label_col not in df.columns:
            label_col = cfg.data.label_col 
            
        self.features = df[num_cols + cat_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)
        
        self.num_classes = len(np.unique(self.labels))
        self.class_names = [str(c) for c in np.unique(self.labels)]
        
        print(f"[{split.upper()}] Caricati {len(df)} campioni.")
        print(f"[{split.upper()}] Feature: {self.features.shape[1]} | Target: {label_col} | Classi: {self.num_classes}")

    def __len__(self): 
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def prepare_loader(cfg, split='train'):
    dataset = NetFlowDataset(cfg, split)
    is_train = (split == 'train')
    
    batch_size = cfg.get('batch_size', cfg.get('experiment', {}).get('batch_size', 32))
    
    # --- CALCOLO AUTOMATICO PESI ---
    labels = dataset.labels
    unique_classes, class_sample_count = np.unique(labels, return_counts=True)
    
    # Pesi per la LOSS (Inversamente proporzionali alla frequenza)
    # Servono per run_linear_probe e run_fine_tuning
    loss_weights = 1. / (class_sample_count.astype(np.float32) + 1e-6)
    loss_weights = loss_weights / loss_weights.sum() * len(unique_classes)
    loss_weights_tensor = torch.from_numpy(loss_weights).float()

    sampler = None
    if is_train:
        # Pesi per il SAMPLER (Smoothing con radice quadrata)
        # Aiuta a bilanciare la composizione dei batch
        sampler_weights = 1. / np.sqrt(class_sample_count) 
        samples_weight = np.array([sampler_weights[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle = False 
    else:
        shuffle = False

    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler, 
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=is_train 
    )
    
    return loader, dataset.class_names, loss_weights_tensor
