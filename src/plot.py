import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm

# ✅ FIX: Import corretto
from src.models import SimCLR
from src.datasets import prepare_loader


def extract_features(model, loader, device):
    """Extracts features from the backbone (h space)."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, target in tqdm(loader, desc="Extracting features"):
            # Compatibility check for multi-view datasets
            if isinstance(imgs, list):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            # .float() is MANDATORY for BF16 compatibility with NumPy
            h = model(imgs, return_features=True).float()
            features.append(h.cpu().numpy())
            labels.append(target.numpy())
            
    return np.concatenate(features), np.concatenate(labels)


def generate_tsne_plot(ckpt_path, base_model='resnet18', experiment_name="", cfg=None):
    """Generates and saves a t-SNE plot from a specific checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    model = SimCLR(base_model=base_model, out_dim=128, num_classes=10).to(device)
    
    # Load Weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✓ Loaded checkpoint: {ckpt_path}")

    # Data Loader (Test set)
    if cfg is None:
        # ⚠️ Se chiamato standalone, crea un config minimale
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({
            'batch_size': 64,
            'data': {
                'root_path': './data',
                'size': 96,
                'num_workers': 4,
                'num_classes': 10,
                'augmentation': {
                    'color_jitter_strength': 0.5,
                    'pin_memory': True
                }
            },
            'experiment': {
                'mode': 'self_supervised',
                'supervised': False
            },
            'seed': 42
        })
    
    test_loader = prepare_loader(cfg, split='test')

    # Feature Extraction
    print("Extracting features...")
    X, y = extract_features(model, test_loader, device)

    print(f"Computing t-SNE for '{experiment_name}'... (may take a few minutes)")
    # Using 'pca' initialization for better global structure
    tsne = TSNE(
        n_components=2, 
        perplexity=30,  
        random_state=42, 
        init='pca', 
        learning_rate='auto'
    )
    X_embedded = tsne.fit_transform(X)

    # Visualization
    plt.figure(figsize=(12, 10))
    class_names = [
        'airplane', 'bird', 'car', 'cat', 'deer', 
        'dog', 'horse', 'monkey', 'ship', 'truck'
    ]
    
    sns.scatterplot(
        x=X_embedded[:, 0], 
        y=X_embedded[:, 1],
        hue=[class_names[i] for i in y],
        palette=sns.color_palette("hls", 10),
        legend="full", 
        alpha=0.6, 
        s=50
    )
    
    plt.title(f"t-SNE: {experiment_name} (STL-10 Test Set)", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    
    # File naming and saving
    save_name = experiment_name.lower().replace(' ', '_')
    save_path = f"tsne_{save_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"✓ Plot saved to: {save_path}")
    plt.close()

    return save_path


if __name__ == "__main__":
    """
    Standalone execution for generating t-SNE plots.
    Usage: python -m src.plot
    """
    print("=== Standalone t-SNE Generation ===\n")
    
    # Example paths (adjust to your actual checkpoint locations)
    checkpoints = [
        ("checkpoints/supervised/last_model.pth", "Supervised SupCon"),
        ("checkpoints/self_supervised/last_model.pth", "Self-Supervised SimCLR"),
    ]
    
    for ckpt_path, exp_name in checkpoints:
        ckpt = Path(ckpt_path)
        if ckpt.exists():
            print(f"\n--- Processing: {exp_name} ---")
            generate_tsne_plot(
                str(ckpt), 
                base_model='resnet18', 
                experiment_name=exp_name
            )
        else:
            print(f"⚠️  Checkpoint not found: {ckpt_path}")
    
    print("\n✓ All t-SNE plots generated!")