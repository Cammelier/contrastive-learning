import hydra
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

# Fix per PyTorch 2.6+ e compatibilità Hydra
torch.serialization.add_safe_globals([DictConfig, ListConfig, dict])

from src.models import SimCLR
from src.datasets import prepare_loader
from src.common.plots import plot_tsne, plot_confusion_matrix

def load_model(cfg, checkpoint_path, device, input_dim_num, cat_dims, num_classes):
    """
    Carica il modello SimCLR inizializzandolo con le dimensioni reali del training set.
    """
    print(f"📂 Caricamento pesi da: {checkpoint_path}")
    
    # Inizializzazione basata sulla struttura definita in src/models.py
    model = SimCLR(
        input_dim_num=input_dim_num,
        cat_dims=cat_dims,
        out_dim=cfg.model.out_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_classes=num_classes
    ).to(device)
    
    # Caricamento checkpoint (weights_only=False per DictConfig)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Mappatura state_dict basata sulle diverse fasi di salvataggio del main.py
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'encoder' in checkpoint:
        state_dict = checkpoint['encoder']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    
    # Caricamento testa di classificazione se presente (per Finetuning o Linear Probe)
    if 'classifier' in checkpoint:
        print("   -> Trovata testa di classificazione. Caricamento...")
        # Se i pesi sono salvati separatamente (Linear Probe)
        if isinstance(checkpoint.get('classifier'), dict):
            model.classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
            model.classifier.load_state_dict(checkpoint['classifier'])
    
    model.eval()
    return model

def run_inference_preds(model, loader, device):
    """
    Esegue l'inferenza sul test set per la Confusion Matrix.
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for (x_num, x_cat), labels in tqdm(loader, desc="Inference Preds", leave=False):
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            
            # Forward pass (Input come tupla come richiesto da src/models.py)
            h, _ = model((x_num, x_cat))
            
            if hasattr(model, 'classifier'):
                logits = model.classifier(h)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
            else:
                return None, None
                
    return np.array(all_labels), np.array(all_preds)

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # 1. Caricamento Train Dataset SOLO per le dimensioni (risolve il size mismatch)
    print("📊 Caricamento Train Dataset per inizializzazione dimensioni...")
    _, train_dataset, _ = prepare_loader(cfg, 'train')
    
    input_dim_num = train_dataset.features_num.shape[1] # 29
    cat_dims = train_dataset.cat_dims # Lista delle cardinalità/embedding
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    
    # 2. Caricamento Test Loader per le inferenze reali
    print("📊 Caricamento Test Loader per generazione grafici...")
    test_loader, _, _ = prepare_loader(cfg, 'test')
    
    plots_dir = root_dir / "thesis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Mappatura dei checkpoint da processare
    checkpoints_to_plot = {
        "SimCLR_Pretrained": root_dir / "checkpoints/simclr/pretrained_encoder.pth",
        "SimCLR_Finetuned":  root_dir / "checkpoints/simclr/best_finetuned.pth",
        "Linear_Probe":      root_dir / "checkpoints/simclr/best_linear_probe.pth",

        # --- Modelli SupCon ---
        "SupCon_Pretrained": root_dir / "checkpoints/supcon/pretrained_encoder.pth",
        "SupCon_Finetuned":  root_dir / "checkpoints/supcon/best_finetuned.pth",
        "SupCon_Probe":      root_dir / "checkpoints/supcon/best_linear_probe.pth"
    }

    for name, path in checkpoints_to_plot.items():
        if not path.exists():
            print(f"⚠️ Saltato: {name} (File non trovato in {path})")
            continue
            
        print(f"\n🎨 Elaborazione: {name}")
        
        try:
            # Inizializzazione sicura con dimensioni da training set
            model = load_model(cfg, path, device, input_dim_num, cat_dims, num_classes)
            
            # A. Generazione t-SNE dello spazio latente (backbone output h)
            print("   -> Generazione t-SNE...")
            plot_tsne(model, test_loader, device, "Final", name, class_names, str(plots_dir))
            
            # B. Generazione Confusion Matrix (solo se il modello è addestrato alla classificazione)
            if "Finetuned" in name or "Probe" in name:
                print("   -> Generazione Confusion Matrix...")
                labels, preds = run_inference_preds(model, test_loader, device)
                if labels is not None:
                    plot_confusion_matrix(labels, preds, class_names, name, str(plots_dir))
                    
        except Exception as e:
            print(f"❌ Errore critico su {name}: {e}")

    print(f"\n✅ Operazione completata. Grafici disponibili in: {plots_dir}")

if __name__ == "__main__":
    main()