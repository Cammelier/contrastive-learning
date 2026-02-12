import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# ← I/O + Preprocessing
from src.data.io import load_df
from src.data.preprocessing import rare_category_filter, TopNCategoryEncoder, ml_split


class NetFlowDataset(Dataset):
    def __init__(self, cfg_data, split='train', seed=42, transform=None):
        # 1. Load dataset 
        df = load_df(f"data/{cfg_data.file_name}.parquet")
        
        
        if 'split' in df.columns:
            df = df[df['split'] == split]  # Filtra per split
            print(f"Loaded {split} split: {len(df)} samples")
        else:
            print(f"No 'split' column, using full dataset: {len(df)} samples")
        
        # 2. Preprocessing minimo 
        df = rare_category_filter(df, cfg_data.cat_cols, cfg_data.min_cat_count)
        df = df.query(cfg_data.filter_query) if cfg_data.filter_query else df
        df = df[df[cfg_data.label_col] != cfg_data.benign_tag]  
        
        print(f"Dataset '{cfg_data.name}' ({split}): {len(df)} samples after filtering")
        
        # 3. Encoding categoriche
        encoder = TopNCategoryEncoder(cfg_data.top_n_categories)
        cat_encoded = encoder.fit_transform(df[cfg_data.cat_cols]) 
        df[cfg_data.cat_cols] = cat_encoded 
        
        # 4. Label Encoding 
        label_col_processed = f"multi_{cfg_data.label_col}"
        if label_col_processed in df.columns:
            labels = df[label_col_processed].values.astype(np.int64)
            self.class_names = ['Benign', 'Attack']  # Da preprocessing
        else:
            le = LabelEncoder()
            labels = le.fit_transform(df[cfg_data.label_col]).astype(np.int64)
            self.class_names = le.classes_
        
       
        if all(col in df.columns for col in cfg_data.num_cols):
            scaler = StandardScaler()
            num_features = scaler.fit_transform(df[cfg_data.num_cols]).astype(np.float32)
        else:
            num_features = np.zeros((len(df), len(cfg_data.num_cols)))  
        
        # 6. Final features
        self.features = np.hstack([
            num_features, 
            df[cfg_data.cat_cols].values.astype(np.float32)
        ])
        self.labels = labels
        
        print(f"Features: {self.features.shape[1]} (29 num + 11 cat)")
        print(f"Classes: {len(self.class_names)}")
        
        self.transform = transform


    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform: x = self.transform(x)
        return x, y


def prepare_loader(cfg, split='train'):
    cfg_data = cfg.data
    dataset = NetFlowDataset(cfg_data, split, cfg.seed)
    
    
    ds = dataset 
    
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=(split=='train'),
        num_workers=cfg.data.num_workers or 0, pin_memory=True
    ), dataset.class_names
