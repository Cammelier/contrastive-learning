import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.io import load_df

class NetFlowDataset(Dataset):
    def __init__(self, cfg, split='train'):
        # 1. Dynamic path
        base_path = Path(cfg.path.processed_data)
        file_path = base_path / f"{cfg.data.file_name}_{split}.{cfg.data.extension}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File non trovato: {file_path}. Esegui prima lo script di preprocessing!")

        # 2. Load file 
        df = load_df(str(file_path))
        
        # 3. Select cols
       
        num_cols = list(cfg.data.num_cols)
        cat_cols = list(cfg.data.cat_cols)
        
   
        label_col = f"multi_{cfg.data.label_col}" 
        if label_col not in df.columns:
            label_col = f"bin_{cfg.data.label_col}" 
            
            
        if label_col not in df.columns:
            label_col = cfg.data.label_col 
            

        # 4. Estrazione Tensor 
        self.features = df[num_cols + cat_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)
        
        
        self.class_names = [str(c) for c in np.unique(self.labels)]
        
        print(f"[{split.upper()}] Caricati {len(df)} campioni. Feature: {self.features.shape[1]}")

    def __len__(self): 
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def prepare_loader(cfg, split='train'):
    """
    Inizializza il dataset e ritorna il DataLoader.
    """
    dataset = NetFlowDataset(cfg, split)
    
   
    is_train = (split == 'train')
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.get('batch_size', cfg.get('experiment', {}).get('batch_size', 32)),
        shuffle=is_train,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=is_train 
    )
    
    return loader, dataset.class_names
