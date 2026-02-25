import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import matplotlib
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path

# Forza il backend non-interattivo
matplotlib.use('Agg')

def extract_features_stratified(model, loader, device, samples_per_class=300, num_classes=10):
    """
    Estrae le feature in modo ottimizzato fermandosi quando ha abbastanza campioni per classe.
    Ideale per dataset da 15M+ di record.
    """
    model.eval()
    features_all = []
    labels_all = []
    
    # Dizionario per contare quanti campioni abbiamo per ogni classe
    counts = {i: 0 for i in range(num_classes)}
    total_needed = samples_per_class * num_classes

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting stratified samples for t-SNE"):
            (x_num, x_cat), target = batch
            
            # Spostiamo al device solo il necessario
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            
            output = model((x_num, x_cat))
            h = output[0] if isinstance(output, tuple) else output
            
            h_np = h.cpu().numpy()
            target_np = target.numpy()

            for i in range(len(target_np)):
                label = int(target_np[i])
                if counts[label] < samples_per_class:
                    features_all.append(h_np[i])
                    labels_all.append(label)
                    counts[label] += 1
            
            # Se abbiamo raggiunto il target per tutte le classi, interrompiamo il caricamento
            if len(labels_all) >= total_needed:
                break
                
    return np.array(features_all), np.array(labels_all)

def plot_tsne(model, loader, device, epoch, mode_name, class_names, save_dir="plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Estrazione ottimizzata (evita di caricare 15M di punti in RAM)
    X, y = extract_features_stratified(model, loader, device, 
                                       samples_per_class=300, 
                                       num_classes=len(class_names))
    
    if len(y) == 0:
        print("⚠️ Errore: Nessun campione estratto per il t-SNE.")
        return

    # Riduzione a 2D con t-SNE
    # n_jobs=-1 usa tutti i core della tua CPU (ne hai 64!)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Creazione etichette testuali sicure
    labels_str = [class_names[i] for i in y]
    
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], 
                    hue=labels_str, 
                    hue_order=class_names, 
                    palette="tab10", 
                    alpha=0.8, 
                    edgecolor='w', 
                    linewidth=0.5)
    
    plt.title(f"t-SNE Visualization: {mode_name} (Epoch {epoch})\nStratified Subsampling (300 samples/class)", fontsize=15)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    filename = Path(save_dir) / f"tsne_{mode_name}_ep{epoch}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ t-SNE salvato con successo: {filename}")

def plot_confusion_matrix(all_labels, all_preds, class_names, mode, save_dir="plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalizzazione per riga (True Labels) per evidenziare le classi minoritarie
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_names, 
                yticklabels=class_names,
                annot_kws={"size": 9})
    
    plt.ylabel('Classe Reale', fontsize=12, fontweight='bold')
    plt.xlabel('Classe Predetta', fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix Normalizzata - {mode}\nDataset: CIC-IDS-2018 (15M samples)", fontsize=15)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    filename = Path(save_dir) / f"cm_{mode}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Matrice di Confusione salvata: {filename}")