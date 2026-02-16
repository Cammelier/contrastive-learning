import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path

def get_stratified_samples(X, y, samples_per_class=500):
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
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting features", leave=False):
            # Gestione tupla NetFlow (x_num, x_cat)
            (x_num, x_cat), target = batch
            
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            
            # Forward pass passando la tupla come nel main.py
            output = model((x_num, x_cat))
            
            if isinstance(output, tuple):
                h = output[0]  # Prende le feature dal backbone
            else:
                h = output
                
            features_list.append(h.cpu().numpy())
            labels_list.append(target.numpy())
            
    return np.concatenate(features_list), np.concatenate(labels_list)

def plot_tsne(model, loader, device, epoch, mode_name, class_names, save_dir="plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    X_full, y_full = extract_features(model, loader, device)
    X, y = get_stratified_samples(X_full, y_full, samples_per_class=300)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    labels_str = [class_names[int(i)] for i in y]
    
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels_str, alpha=0.7)
    plt.title(f"t-SNE: {mode_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    filename = Path(save_dir) / f"tsne_{mode_name}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ t-SNE salvato in: {filename}")

def plot_confusion_matrix(all_labels, all_preds, class_names, mode, save_dir="plots"):
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {mode}")
    
    filename = Path(save_dir) / f"cm_{mode}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ CM salvata in: {filename}")