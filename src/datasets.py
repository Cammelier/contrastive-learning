import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ← I/O + Preprocessing
from src.data.io import load_df
from src.data.preprocessing import rare_category_filter, TopNCategoryEncoder, ml_split

class NetFlowDataset(Dataset):
    def __init__(self, cfg_data, split='train', seed=42, transform=None):
        # 1. Load dataset 
        file_path = f"data/{cfg_data.file_name}.parquet"
        df = load_df(file_path)
        
        if 'split' in df.columns:
            df = df[df['split'] == split].copy() # .copy() evita problemi di SettingWithCopyWarning
            print(f"Loaded {split} split: {len(df)} samples")
        else:
            print(f"No 'split' column, using full dataset: {len(df)} samples")
        
        # Protezione: se il DataFrame è vuoto, i passaggi successivi falliranno
        if df.empty:
            raise ValueError(f"Il DataFrame caricato per lo split '{split}' è vuoto. Controlla il file {file_path}")

        # 2. Encoding categoriche
        # Usiamo .loc e .values per risolvere il ValueError: Cannot set a DataFrame with multiple columns
        cat_cols = list(cfg_data.cat_cols)
        encoder = TopNCategoryEncoder(cfg_data.top_n_categories)
        cat_encoded = encoder.fit_transform(df[cat_cols]) 
        
        # Assegnazione robusta
        df.loc[:, cat_cols] = cat_encoded.values if hasattr(cat_encoded, 'values') else cat_encoded
        
        # 3. Label Encoding 
        label_col_processed = f"multi_{cfg_data.label_col}"
        if label_col_processed in df.columns:
            labels = df[label_col_processed].values.astype(np.int64)
            # Recuperiamo i nomi classi se disponibili, altrimenti default
            self.class_names = ['Benign', 'Attack'] 
        else:
            le = LabelEncoder()
            labels = le.fit_transform(df[cfg_data.label_col]).astype(np.int64)
            self.class_names = le.classes_.tolist()
        
        # 4. Numerical Scaling
        num_cols = list(cfg_data.num_cols)
        if all(col in df.columns for col in num_cols):
            scaler = StandardScaler()
            num_features = scaler.fit_transform(df[num_cols]).astype(np.float32)
        else:
            print(f"Warning: Alcune colonne numeriche mancano. Uso zeri.")
            num_features = np.zeros((len(df), len(num_cols)), dtype=np.float32)
        
        # 5. Final features
        # Concateniamo numeriche e categoriche (già codificate)
        self.features = np.hstack([
            num_features, 
            df[cat_cols].values.astype(np.float32)
        ])
        self.labels = labels
        self.labels = (self.labels > 0).astype(np.int64) 
        self.class_names = ['Benign', 'Attack']
        
        print(f"Features: {self.features.shape[1]} ({len(num_cols)} num + {len(cat_cols)} cat)")
        print(f"Classes: {len(self.class_names)}")
        
        self.transform = transform

    def __len__(self): 
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform: 
            x = self.transform(x)
        return x, y

def prepare_loader(cfg, split='train'):
    cfg_data = cfg.data
    dataset = NetFlowDataset(cfg_data, split, cfg.seed)
    
    return DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        shuffle=(split=='train'),
        num_workers=cfg.data.num_workers or 0, 
        pin_memory=True
    ), dataset.class_names
