import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path

# --- FUNZIONI DI SUPPORTO ---

def get_stratified_samples(X, y, samples_per_class=500):
    """Estrae un numero bilanciato di campioni per ogni classe (per t-SNE leggibile)."""
    stratified_indices = []
    unique_classes = np.unique(y)
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_samples = min(len(cls_indices), samples_per_class)
        if n_samples > 0:
            selected = np.random.choice(cls_indices, n_samples, replace=False)
            stratified_indices.extend(selected)
    
    return X[stratified_indices], y[stratified_indices]

def extract_features(model, loader, device):
    """Estrae le feature (h) dal modello."""
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features", leave=False):
            if isinstance(batch, (list, tuple)):
                x, target = batch[0], batch[1]
            else:
                x, target = batch, torch.zeros(len(batch))
            
            x = x.to(device)
            # Gestione output SimCLR (tuple) vs Linear Probe (tensor)
            output = model(x)
            if isinstance(output, tuple):
                h = output[0] # SimCLR restituisce (h, z)
            else:
                h = output    # Linear Probe/Finetuned restituisce logits o features
                
            features_list.append(h.cpu().numpy())
            labels_list.append(target.numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)

# --- GRAFICO 1: t-SNE (Spazio Latente) ---

def plot_tsne(model, loader, device, epoch, mode_name, class_names, save_dir="plots"):
    """Genera e salva il plot t-SNE."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🎨 Generazione t-SNE per {mode_name}...")
    X_full, y_full = extract_features(model, loader, device)
    
    # Campionamento Stratificato (Cruciale per classi rare)
    X, y = get_stratified_samples(X_full, y_full, samples_per_class=300)
    
    # Calcolo t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', learning_rate='auto', random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")
    
    unique_y = np.unique(y)
    # Assicura che la palette copra tutte le classi
    palette = sns.color_palette("tab10", len(class_names)) if len(class_names) <= 10 else sns.color_palette("husl", len(class_names))
    
    # Mapping sicuro degli indici alle label stringa
    labels_str = [class_names[int(i)] for i in y]
    
    sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=labels_str,
        palette=palette,
        alpha=0.7, s=50, edgecolor='k', linewidth=0.1
    )
    
    plt.title(f"t-SNE Visualization: {mode_name} (Epoch {epoch})", fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Classes")
    plt.tight_layout()
    
    filename = f"{save_dir}/tsne_{mode_name}_ep{epoch}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    if wandb.run is not None:
        wandb.log({f"tsne/{mode_name}": wandb.Image(filename)})
    
    print(f"✅ t-SNE salvato: {filename}")

# --- GRAFICO 2: Confusion Matrix (Risultati) ---

def plot_confusion_matrix(all_labels, all_preds, class_names, mode, save_dir="plots"):
    """Genera e salva la matrice di confusione normalizzata."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    # Normalizzazione per riga (Recall per classe)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Gestisce divisioni per zero se una classe manca
    
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.0)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Recall (Normalized by True Labels)'})
    
    plt.title(f"Confusion Matrix - {mode.upper()}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f"{save_dir}/cm_{mode}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    if wandb.run is not None:
        wandb.log({f"test/{mode}_cm": wandb.Image(filename)})
        
    print(f"✅ CM salvata: {filename}")