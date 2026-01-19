import hydra 
import torch 
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 

from src.datasets import get_stl10_dataloader
from src.models import SimCLR
from src.losses import ContrastiveLoss


@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup path and environment
    root_dir = Path(hydra.utils.get_original_cwd())
   
    ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # 2. Wandb initialization 
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.logger.project,
        group=cfg.experiment.mode,
        name=f"{cfg.experiment.mode}_seed{cfg.seed}",
        config=wandb_config
    )

    print(f"--- Avvio Esperimento: {cfg.experiment.mode} ---")
    print(f"Device: {device} | Cartella checkpoint: {ckpt_dir}")

    # 3. DataLoader 
    if cfg.experiment.supervised:
        # Per supervised: usa split 'train' con label
        train_loader = get_stl10_dataloader(cfg, split='train')
        val_loader = get_stl10_dataloader(cfg, split='test')
    else:
        # Per self-supervised: usa 'unlabeled'
        train_loader = get_stl10_dataloader(cfg, split='unlabeled')
        val_loader = None

    # 4. Model, criterion, optimizer
    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)

    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=cfg.experiment.supervised
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-4
    )

    # 5. Linear classifier (solo per supervised)
    classifier = None
    classifier_optimizer = None
    if cfg.experiment.supervised:
        feature_dim = 512  # ResNet18 output dimension
        num_classes = cfg.data.num_classes
        
        classifier = nn.Linear(feature_dim, num_classes).to(device)
        classifier_optimizer = torch.optim.Adam(
            classifier.parameters(), 
            lr=cfg.learning_rate
        )
        print(f"Modalità Supervised: Linear classifier aggiunto ({feature_dim} -> {num_classes})")

    # 6. Training loop
    for epoch in range(cfg.epochs):
        model.train()
        if classifier is not None:
            classifier.train()
        
        total_loss = 0
        total_cls_loss = 0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{cfg.epochs}")
        
        for batch_idx, (imgs, labels) in enumerate(pbar):
            x_i, x_j = imgs[0].to(device), imgs[1].to(device)
            labels = labels.to(device)

            # Forward pass attraverso encoder
            x_combined = torch.cat([x_i, x_j], dim=0)
            h_combined, z_combined = model(x_combined)
            
            # Split features
            z_i, z_j = torch.split(z_combined, x_i.size(0))
            features = torch.stack([z_i, z_j], dim=1)

            # Contrastive loss
            if cfg.experiment.supervised:
                loss = criterion(features, labels)
            else:
                loss = criterion(features, None)

            # Backward per encoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Supervised: train classifier e calcola accuracy
            if cfg.experiment.supervised:
                # Usa features dalla prima view (h_i)
                h_i, _ = torch.split(h_combined, x_i.size(0))
                
                # Forward attraverso classifier
                logits = classifier(h_i.detach())  # detach per non fare backprop nell'encoder
                cls_loss = nn.functional.cross_entropy(logits, labels)
                
                # Backward per classifier
                classifier_optimizer.zero_grad()
                cls_loss.backward()
                classifier_optimizer.step()
                
                # Calcola accuracy
                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                
                total_cls_loss += cls_loss.item()
                
                # Accuracy corrente del batch
                batch_acc = correct / labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'cont_loss': f'{loss.item():.4f}',
                    'cls_loss': f'{cls_loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
                
                # Log su wandb ogni 20 batch
                if batch_idx % 20 == 0:
                    wandb.log({
                        "batch/contrastive_loss": loss.item(),
                        "batch/classifier_loss": cls_loss.item(),
                        "batch/accuracy": batch_acc
                    })
            else:
                # Self-supervised: solo contrastive loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                if batch_idx % 20 == 0:
                    wandb.log({"batch/contrastive_loss": loss.item()})

        # Metriche di fine epoca
        avg_loss = total_loss / len(train_loader)
        
        log_dict = {
            "epoch": epoch + 1,
            "epoch/avg_contrastive_loss": avg_loss,
            "epoch/learning_rate": optimizer.param_groups[0]['lr']
        }
        
        if cfg.experiment.supervised:
            avg_cls_loss = total_cls_loss / len(train_loader)
            train_accuracy = total_correct / total_samples
            
            log_dict["epoch/avg_classifier_loss"] = avg_cls_loss
            log_dict["epoch/train_accuracy"] = train_accuracy
            
            print(f"Epoch [{epoch+1}/{cfg.epochs}] | "
                  f"Cont Loss: {avg_loss:.4f} | "
                  f"Cls Loss: {avg_cls_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{cfg.epochs}] | Avg Loss: {avg_loss:.4f}")
        
        wandb.log(log_dict)

        # Validation (solo per supervised)
        if cfg.experiment.supervised and val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            classifier.eval()
            
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validation", leave=False):
                    x = imgs[0].to(device)
                    labels = labels.to(device)
                    
                    h, _ = model(x)
                    logits = classifier(h)
                    
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
                    
                    loss = nn.functional.cross_entropy(logits, labels)
                    val_loss += loss.item()
            
            val_accuracy = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            wandb.log({
                "validation/accuracy": val_accuracy,
                "validation/loss": avg_val_loss,
                "epoch": epoch + 1
            })
            
            print(f"  Validation Acc: {val_accuracy:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            model.train()
            classifier.train()
    
        # Save checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.epochs:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch+1}.pth"
            
            save_dict = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "contrastive_loss": avg_loss,
            }
            
            if cfg.experiment.supervised:
                save_dict["classifier_state_dict"] = classifier.state_dict()
                save_dict["classifier_optimizer_state_dict"] = classifier_optimizer.state_dict()
                save_dict["train_accuracy"] = train_accuracy
            
            torch.save(save_dict, ckpt_path)
            print(f"Checkpoint salvato: {ckpt_path}")
    
    wandb.finish()
    print("--- Esperimento Terminato ---")

if __name__ == "__main__":
    main()