import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.io import load_df
from sklearn.preprocessing import StandardScaler

class NetFlowDataset(Dataset):
    def __init__(self, cfg, split='train'):
        base_path = Path(cfg.path.processed_data)
        file_path = base_path / f"{cfg.data.file_name}_{split}.{cfg.data.extension}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}")

        # load_df legge automaticamente .parquet se l'estensione è corretta
        df = load_df(str(file_path))
        
        num_cols = list(cfg.data.num_cols)
        cat_cols = list(cfg.data.cat_cols)
        
        label_col = f"multi_{cfg.data.label_col}" 
        if label_col not in df.columns:
            label_col = cfg.data.label_col 
            
        # 1. Feature Numeriche
        features_num = df[num_cols].values.astype(np.float32)
        features_num = np.log1p(np.maximum(features_num, 0))
        scaler = StandardScaler()
        self.features_num = scaler.fit_transform(features_num)
    
        # 2. Feature Categoriche
        self.features_cat = df[cat_cols].values.astype(np.int64) 
    
        # Calcolo dimensioni per gli embedding
        self.cat_dims = []
        for col_idx in range(self.features_cat.shape[1]):
            n_cat = int(self.features_cat[:, col_idx].max() + 1)
            emb_dim = min(50, (n_cat + 1) // 2)
            self.cat_dims.append((n_cat, emb_dim))

        self.labels = df[label_col].values.astype(np.int64)
        self.num_classes = len(np.unique(self.labels))
        self.class_names = ["Benign", "Bot", "Brute Force -Web", "Brute Force -XSS", "DDOS attack-HOIC","DDOS attack-LOIC-UDP", "DoS attacks-GoldenEye","DoS attacks-Hulk", "DoS attacks-SlowHTTPTest",   "DoS attacks-Slowloris", "Infiltration"]
        
        print(f"[{split.upper()}] Caricati {len(df)} campioni.")
        # FIX: Usiamo features_num per la stampa invece di features
        print(f"[{split.upper()}] Feature Numeriche: {self.features_num.shape[1]} | Categoriche: {self.features_cat.shape[1]}")

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        x_num = torch.from_numpy(self.features_num[idx])
        x_cat = torch.from_numpy(self.features_cat[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return (x_num, x_cat), y

def prepare_loader(cfg, split='train'):
    dataset = NetFlowDataset(cfg, split)
    is_train = (split == 'train')
    
    batch_size = cfg.get('batch_size', cfg.experiment.get('batch_size', 512))
    
    labels = dataset.labels
    unique_classes, class_sample_count = np.unique(labels, return_counts=True)
    
    loss_weights = 1. / (np.power(class_sample_count, 0.25) + 1e-6)
      
    loss_weights = loss_weights / loss_weights.sum() * len(unique_classes)
    loss_weights_tensor = torch.from_numpy(loss_weights).float()

    sampler = None
    if is_train:
        sampler_weights = 1. / class_sample_count
        samples_weight = torch.from_numpy(sampler_weights[labels]).double()
        
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
    
    return loader, dataset, loss_weights_tensor
