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
    
    # Prendi il batch_size corretto in base alla fase
    if is_train and 'experiment' in cfg:
        batch_size = cfg.experiment.get('batch_size', 256)
    else:
        batch_size = cfg.get('batch_size', 256)
    
    labels = dataset.labels
    unique_classes, class_sample_count = np.unique(labels, return_counts=True)
    
    # Calcolo pesi per la CrossEntropyLoss (Finetuning)
    loss_weights = 1. / (class_sample_count.astype(np.float32) + 1e-6)
    loss_weights_tensor = torch.from_numpy(loss_weights).float()

    sampler = None
    if is_train:
        # Pesi per il WeightedRandomSampler
        sampler_weights = 1. / class_sample_count # Più aggressivo per SupCon
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
        num_workers=cfg.get('system', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=is_train 
    )
    
    # Ritorna anche il dataset per il pre-training
    return loader, dataset, loss_weights_tensor

