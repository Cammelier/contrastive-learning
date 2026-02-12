import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class NetFlowDataset(Dataset):
    def __init__(self, name, filename, extension='parquet', label_col='Attack', 
                 num_cols=None, cat_cols=None, benigntag='Benign', 
                 trainfrac=0.8, valfrac=0.1, testfrac=0.1, 
                 filter_query=None, mincatcount=100, topncategories=32,
                 seed=42, transform=None):
        
        
        data_path = f"data/{filename}.{extension}"
        if extension == 'parquet':
            df = pq.read_table(data_path).to_pandas()
        else:
            df = pd.read_csv(data_path)
        
        #
        if filter_query:
            df = df.query(filter_query)
        if benigntag:
            df = df[df[label_col] != benigntag]
        
        print(f"Dataset '{name}': {len(df)} samples after filtering")
        
        # 🎯 Encoding label
        le = LabelEncoder()
        self.labels = le.fit_transform(df[label_col]).astype(np.int64)
        self.class_names = le.classes_
        self.num_classes = len(self.class_names)
        
        # features → StandardScaler
        if num_cols:
            self.num_scaler = StandardScaler()
            df[num_cols] = self.num_scaler.fit_transform(df[num_cols])
            num_features = df[num_cols].values.astype(np.float32)
        else:
            num_features = np.empty((len(df), 0))
        
        # CATEGORICAL → Top-N encoding
        cat_features = []
        if cat_cols:
            for col in cat_cols:
                vc = df[col].value_counts()
                top_cats = vc.head(topncategories).index
                # Mappa a indici 0..top_n-1, resto → 0
                mapping = {cat: i+1 for i, cat in enumerate(top_cats)}
                df[f"{col}_enc"] = df[col].map(mapping).fillna(0).astype(np.int64)
                if col in vc[vc >= mincatcount].index:
                    cat_features.append(f"{col}_enc")
        
        # Final features = num + cat_encoded
        if cat_features:
            self.feature_cols = num_cols + cat_features
            cat_part = df[cat_features].values.astype(np.float32)
            self.features = np.hstack([num_features, cat_part])
        else:
            self.feature_cols = num_cols
            self.features = num_features
        
        print(f"Features: {self.features.shape[1]} (num:{num_features.shape[1]}, cat:{len(cat_features)})")
        print(f"Classes: {self.num_classes}")
        
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
    """Carica da YAML Hydra."""
    cfg_data = cfg.data
    
    # Craete dataset 
    full_dataset = NetFlowDataset(
        name=cfg_data.name,
        filename=cfg_data.file_name,
        extension=cfg_data.extension,
        label_col=cfg_data.label_col,
        num_cols=cfg_data.numcols,
        cat_cols=cfg_data.catcols,
        benigntag=cfg_data.benigntag,
        trainfrac=cfg_data.trainfrac,
        valfrac=cfg_data.valfrac,
        testfrac=cfg_data.testfrac,
        filter_query=getattr(cfg_data, 'filter_query', None),
        mincatcount=getattr(cfg_data, 'mincatcount', 100),
        topncategories=getattr(cfg_data, 'topncategories', 32),
        seed=cfg.seed
    )
    
    
    lengths = [cfg_data.trainfrac, cfg_data.valfrac, cfg_data.testfrac]
    lengths = [int(len(full_dataset) * f) for f in lengths]
    lengths[-1] = len(full_dataset) - sum(lengths[:-1])  # adjust test
    
    g = torch.Generator().manual_seed(cfg.seed)
    splits = random_split(full_dataset, lengths, generator=g)
    
    datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
    
    # Loader
    if split == 'train' or split == 'unlabeled':
        ds = datasets['train']
        shuffle, drop_last = True, True
    elif split == 'val':
        ds = datasets['val']
        shuffle, drop_last = False, False
    else:
        ds = datasets['test']
        shuffle, drop_last = False, False
    
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.data.num_workers, pin_memory=True, drop_last=drop_last
    )
    
    return loader, full_dataset.class_names

