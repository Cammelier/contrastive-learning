import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from tqdm import tqdm

# Importa le classi del tuo progetto
from models import SimCLR 
from data_loader import prepare_loader

def extract_features(model, loader, device):
    """Extracts features from the backbone (h space)."""
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for imgs, target in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device)
            # return_features=True returns the output of the backbone (512 or 2048 dim)
            h = model(imgs, return_features=True)
            features.append(h.cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(features), np.concatenate(labels)

def generate_tsne_plot(ckpt_path, base_model='resnet18', experiment_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model (ensure same params as training)
    # Using 10 classes for STL-10
    model = SimCLR(base_model=base_model, out_dim=128, num_classes=10).to(device)
    
    # 2. Load Weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {ckpt_path}")

    # 3. Data Loader (Test set - standard normalization)
    # We use batch_size 256 for faster extraction
    test_loader = prepare_loader(split='test', batch_size=256)

    # 4. Feature Extraction
    X, y = extract_features(model, test_loader, device)

    # 5. t-SNE Projection
    print(f"Computing t-SNE for {experiment_name}... (this may take a few minutes)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # 6. Visualization
    plt.figure(figsize=(12, 10))
    class_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=[class_names[i] for i in y],
        palette=sns.color_palette("hls", 10),
        legend="full", alpha=0.6, s=50
    )
    
    plt.title(f"t-SNE: {experiment_name} (STL-10 Test Set)", fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    
    # Save high-quality image for the thesis
    save_path = f"tsne_{experiment_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURE THESE PATHS FOR YOUR THESIS ---
    
    sup_ckpt = "outputs/supervised_run/last_model.pth" 
    generate_tsne_plot(sup_ckpt, base_model='resnet18', experiment_name="Supervised Contrastive")

    ssl_ckpt = "outputs/ssl_run/last_model.pth"
    generate_tsne_plot(ssl_ckpt, base_model='resnet18', experiment_name="Self-Supervised SimCLR")
