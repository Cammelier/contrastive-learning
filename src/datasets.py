import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # <--- Aggiunto Sampler
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.io import load_df

class NetFlowDataset(Dataset):
    def __init__(self, cfg, split='train'):
        base_path = Path(cfg.path.processed_data)
        file_path = base_path / f"{cfg.data.file_name}_{split}.{cfg.data.extension}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}. Esegui prima lo script di preprocessing!")

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
    
    sampler = None
    # BILANCIAMENTO: Applichiamo il campionamento pesato solo in fase di training/probe
    if is_train:
        labels = dataset.labels
        class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
        
        # Il peso è l'inverso della frequenza (più rara è la classe, più alto è il peso)
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight).double()
        
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        # Nota: Quando si usa il sampler, 'shuffle' deve essere False
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
    
    return loader, dataset.class_names
