print("🟢 SCRIPT AVVIATO: Inizio importazione moduli...") # DEBUG PRINT

import hydra
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, ListConfig

# Fix per PyTorch 2.6+
torch.serialization.add_safe_globals([DictConfig, ListConfig, dict])

# Import dei tuoi moduli (Assicurati che src/common/plots.py sia quello che ti ho mandato prima!)
from src.models import SimCLR
from src.datasets import prepare_loader
from src.common.plots import plot_tsne, plot_confusion_matrix

print("🟢 MODULI IMPORTATI. Configurazione funzione Load Model...") # DEBUG PRINT

# --- FUNZIONI LOCALI ---

def load_model(cfg, checkpoint_path, device):
    print(f"📂 Caricamento pesi da: {checkpoint_path}")
    
    # Istanziazione Modello Base
    model = SimCLR(
        input_dim=cfg.data.input_dim, 
        hidden_dim=cfg.model.hidden_dim, 
        out_dim=cfg.model.out_dim
    ).to(device)
    
    # Caricamento Checkpoint (Con fix per il bug UnpicklingError)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Estrazione pesi corretti
    if 'model' in checkpoint: state_dict = checkpoint['model']
    elif 'encoder' in checkpoint: state_dict = checkpoint['encoder']
    elif 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    else: state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    
    # Se c'è un classificatore (per confusion matrix), carichiamolo
    if 'classifier' in checkpoint:
        print("   -> Trovata testa di classificazione. Caricamento...")
        num_classes = 9 
        if 'class_names' in checkpoint: num_classes = len(checkpoint['class_names'])
        
        model.classifier = nn.Linear(cfg.model.hidden_dim, num_classes).to(device)
        model.classifier.load_state_dict(checkpoint['classifier'])
    
    model.eval()
    return model

def run_inference_preds(model, loader, device):
    """Serve solo per la Confusion Matrix"""
    if not hasattr(model, 'classifier'): return None, None
    results, targets = [], []
    
    # Import tqdm qui per sicurezza
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference Preds"):
            if isinstance(batch, (list, tuple)): x, y = batch[0], batch[1]
            else: x, y = batch, torch.zeros(len(batch))
            
            x = x.to(device)
            features = model(x)
            # Gestione output tupla di SimCLR
            if isinstance(features, tuple): features = features[0]
            
            logits = model.classifier(features)
            preds = logits.argmax(1)
            results.append(preds.cpu().numpy())
            targets.append(y.numpy())
            
    return np.concatenate(results), np.concatenate(targets)


# --- MAIN ---

@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("🚀 MAIN PARTITO. Inizializzazione...") # DEBUG PRINT
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # Cartella Output
    plots_dir = root_dir / "thesis_plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Dataset
    print("📊 Preparazione Test Loader...")
    test_loader, dataset, _ = prepare_loader(cfg, split='test')
    class_names = getattr(dataset, 'class_names', [str(i) for i in range(9)])
    
    # Lista Checkpoint da processare
    checkpoints_to_plot = {
        "SimCLR_Pretrained": root_dir / "checkpoints/simclr/pretrained_encoder.pth",
        "SupCon_Pretrained": root_dir / "checkpoints/supcon/pretrained_encoder.pth",
        "SimCLR_Finetuned":  root_dir / "checkpoints/simclr/best_finetuned.pth",
        "SupCon_Finetuned":  root_dir / "checkpoints/supcon/best_finetuned.pth"
    }

    for name, path in checkpoints_to_plot.items():
        if not path.exists():
            print(f"⚠️ FILE MANCANTE: {name} (Cercato in: {path})")
            continue
            
        print(f"\n------------------------------------------------")
        print(f"🎨 ELABORAZIONE: {name}")
        print(f"------------------------------------------------")
        
        try:
            model = load_model(cfg, path, device)
            
            # 1. t-SNE (Sempre)
            # Chiama la funzione che hai sistemato in src/common/plots.py
            plot_tsne(model, test_loader, device, "FINAL", name, class_names, str(plots_dir))
            
            # 2. Confusion Matrix (Solo se finetuned)
            if "Finetuned" in name:
                print("   -> Calcolo predizioni per Confusion Matrix...")
                preds, labels = run_inference_preds(model, test_loader, device)
                if preds is not None:
                    plot_confusion_matrix(labels, preds, class_names, name, str(plots_dir))
                else:
                    print("   -> Skip CM: Nessun classificatore trovato nel modello.")
                    
        except Exception as e:
            print(f"❌ ERRORE CRITICO SU {name}: {e}")
            import traceback
            traceback.print_exc() # Stampa l'errore completo per capire cosa non va

    print(f"\n✅ SCRIPT COMPLETATO. Controlla la cartella: {plots_dir}")

# --- PUNTO DI INGRESSO (IMPORTANTE!) ---
if __name__ == "__main__":
    main()