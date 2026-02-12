import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ← I/O + Preprocessing
from src.data.io import load_df
from src.data.preprocessing import rare_category_filter, TopNCategoryEncoder, ml_split

class NetFlowDataset(Dataset):
    def __init__(self, cfg_data, split='train', seed=42, transform=None):
        # 1. Carica Parquet con io.py (cic_2018_v2.parquet)
        df = load_df(f"data/{cfg_data.file_name}.parquet")
        
        # 2. Preprocessing pipeline
        df = rare_category_filter(df, cfg_data.cat_cols, cfg_data.min_cat_count)
        df = df.query(cfg_data.filter_query) if cfg_data.filter_query else df
        df = df[df[cfg_data.label_col] != cfg_data.benign_tag]  
        
        print(f"Dataset '{cfg_data.name}': {len(df)} samples after filtering")
        
        # 3. Encoding categoriche (TopN da preprocessing.py)
        encoder = TopNCategoryEncoder(cfg_data.top_n_categories)
        df[cfg_data.cat_cols] = encoder.fit_transform(df[cfg_data.cat_cols])
        
        # 4. Label Encoding
        le = LabelEncoder()
        labels = le.fit_transform(df[cfg_data.label_col]).astype(np.int64)
        self.class_names = le.classes_
        
        # 5. Numeriche + StandardScaler
        scaler = StandardScaler()
        num_features = scaler.fit_transform(df[cfg_data.num_cols]).astype(np.float32)
        
        # 6. Final features
        self.features = np.hstack([
            num_features, 
            df[cfg_data.cat_cols].values.astype(np.float32)
        ])
        self.labels = labels
        
        print(f"Features: {self.features.shape[1]} (29 num + 11 cat)")
        print(f"Classes: {len(self.class_names)}")
        
        # 7. Split 80/10/10
        self.train_df, self.val_df, self.test_df = ml_split(
            df.assign(features=self.features, labels=labels),
            cfg_data.train_frac, cfg_data.val_frac, cfg_data.test_frac,
            cfg_data.label_col, seed
        )
        
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
    
    
    if split == 'train':
        ds = dataset.train_ds
    elif split == 'val':
        ds = dataset.val_ds
    else:
        ds = dataset.test_ds
        
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=(split=='train'),
        num_workers=cfg.data.num_workers, pin_memory=True
    ), dataset.class_names
