import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.models import SimCLR
from src.datasets import prepare_loader  

def extract_features(model, loader, device):
    """Extracts features from backbone (h space)"""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for x, target in tqdm(loader, desc="Extracting features"):  
            x = x.to(device)
            h = model(x, return_features=True).float()
            features.append(h.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features), np.concatenate(labels)

def generate_tsne_plot(ckpt_path, input_dim=78, experiment_name="", cfg=None):
    """t-SNE for NetFlow contrastive representations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = SimCLR(
        input_dim=input_dim,  
        hidden_dim=512, 
        out_dim=128, 
        num_classes=None  
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✓ Loaded NetFlow checkpoint: {ckpt_path}")

    
    test_loader, class_names = prepare_loader(cfg, split='test')
    print(f"Classes: {class_names}")

    # Feature extraction
    X, y = extract_features(model, test_loader, device)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)//10), 
                random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=[class_names[i] for i in y],
        palette=sns.color_palette("hls", len(class_names)),
        legend="full", alpha=0.6, s=30
    )
    
    plt.title(f"t-SNE NetFlow: {experiment_name}", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_name = experiment_name.lower().replace(' ', '_')
    save_path = Path(f"tsne_netflow_{save_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path}")
    return str(save_path)

def plot_enhanced_confusion_matrix(all_labels, all_preds, class_names, mode):
    cm = confusion_matrix(all_labels, all_preds)
    # Normalizziamo per riga per vedere la precisione percentuale di ogni classe
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Normalized Confusion Matrix - {mode.upper()}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Salva il grafico per la tesi
    plt.savefig(f"confusion_matrix_{mode}.png", dpi=300)
    # Log su WandB
    wandb.log({f"test/{mode}_cm_image": wandb.Image(plt)})
    plt.close()

if __name__ == "__main__":
    # Esempio per cic_2018_v2
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        'data': {
            'csv_path': './data/cic_2018_v2.csv',
            'feature_cols': [...], 
            'label_col': 'label',
            'batch_size': 512
        }
    })
    
    generate_tsne_plot(
        "checkpoints/self_supervised/last_model.pth",
        input_dim=78,
        experiment_name="CIC-2018 Self-Supervised",
        cfg=cfg
    )

