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

def generate_tsne_plot(ckpt_path, input_dim=40, experiment_name="SupCon_NB15", cfg=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Inizializzazione Modello coerente con il tuo script
    model = SimCLR(input_dim=input_dim, hidden_dim=512).to(device)
    
    # 2. Caricamento Checkpoint (Corretto per PyTorch 2.6 e chiavi attuali)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Usiamo la chiave 'model' come salvato nel tuo run_pretraining
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f"✓ Checkpoint caricato correttamente")

    # 3. Loader con sottocampionamento per t-SNE (Max 5000 campioni per leggibilità)
    test_loader, dataset, _ = prepare_loader(cfg, split='test')
    class_names = dataset.class_names
    
    # Estraiamo un numero limitato di campioni per evitare crash di memoria
    print("⏳ Estrazione feature in corso...")
    X_full, y_full = extract_features(model, test_loader, device)
    
    indices = np.random.choice(len(X_full), min(5000, len(X_full)), replace=False)
    X, y = X_full[indices], y_full[indices]

    # 4. t-SNE SOTA Parameters
    print(f"🚀 Calcolo t-SNE su {len(X)} campioni...")
    tsne = TSNE(
        n_components=2, 
        perplexity=40, # Alzato per cluster più definiti
        random_state=42, 
        init='pca', 
        learning_rate='auto',
        n_iter=1000
    )
    X_embedded = tsne.fit_transform(X)

    # 5. Plotting con stile professionale
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    # Creiamo una palette distinta per le 9 classi
    palette = sns.color_palette("bright", len(class_names))
    
    scatter = sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=[class_names[i] for i in y],
        palette=palette,
        legend="full", alpha=0.7, s=40, edgecolor='w', linewidth=0.5
    )
    
    plt.title(f"Visualizzazione Spazio Latente: {experiment_name}\n(UNSW-NB15 Multiclass)", fontsize=16)
    plt.legend(title="Classi di Attacco", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    save_path = Path(f"tsne_{experiment_name.lower()}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot salvato in: {save_path}")
    plt.show()


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

