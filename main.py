import hydra 
import torch 
import torch.nn as nn
import wandb
import sys
from torchmetrics.functional import accuracy 
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 
from src.datasets import prepare_loader
from src.models import SimCLR
from src.losses import ContrastiveLoss

def run_validation(model, classifier, val_loader, criterion, device, cfg):
    
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            # Input 
            if isinstance(imgs, list):
                x_i, x_j = imgs[0].to(device), imgs[1].to(device)
            else:
                x_i = imgs.to(device)
                x_j = x_i
            
            labels = labels.to(device)
            if cfg.experiment.supervised:
                labels = labels - 1
            # 1. Contrastive Loss 
            x_combined = torch.cat([x_i, x_j], dim=0)
            h_combined, z_combined = model(x_combined)
            z_i, z_j = torch.split(z_combined, x_i.size(0))
            
            loss = criterion(torch.cat([z_i, z_j], dim=0), labels if cfg.experiment.supervised else None)
            val_loss += loss.item()

            # 2. Accuracy (Linear Probing online)
            if labels.min() >= 0:
                h_i, _ = torch.split(h_combined, x_i.size(0))
                logits = classifier(h_i) # No detach necessario in eval()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    avg_acc = (val_correct / val_samples) if val_samples > 0 else 0
    return avg_loss, avg_acc

def run_training(cfg, device, model, ckpt_dir):
    # 1. DataLoader 
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.supervised else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')

    # 2. Criterion & Optimizer
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=cfg.experiment.supervised
    ).to(device)

    accumlation_steps = cfg.experiment.get("accumlation_steps", 4)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=cfg.experiment.epochs
)

    # 3. Classifier (Online Linear Probing)
    classifier = nn.Linear(512, 10).to(device)
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.experiment.learning_rate)

    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoca {epoch+1}/{cfg.experiment.epochs}")
        for i,(imgs, labels) in enumerate(pbar):
            
            if isinstance(imgs, list):
                x_i, x_j = imgs[0].to(device), imgs[1].to(device)
            else: 
                x_i = imgs.to(device)
                x_j = x_i 
            
            labels = labels.to(device)

            
            is_supervised = cfg.experiment.supervised
            if is_supervised:
                labels = labels - 1 

            # Contrastive Learning 
            x_combined = torch.cat([x_i, x_j], dim=0)
            h_combined, z_combined = model(x_combined)
            z_i, z_j = torch.split(z_combined, x_i.size(0))
            
            # contrastive loss calculation 
            loss = criterion(torch.cat([z_i, z_j], dim=0), labels if is_supervised else None)
            scaled_loss = loss / accumlation_steps
            scaled_loss.backward()
           
            if (i+1) % accumlation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        

            # --- STEP 2: Linear Evaluation  ---
            
            if labels.min() >= 0:
                h_i, _ = torch.split(h_combined, x_i.size(0))
                logits = classifier(h_i.detach())
                cls_loss = nn.functional.cross_entropy(logits, labels)
                
                cls_optimizer.zero_grad()
                cls_loss.backward()
                cls_optimizer.step()

                # Accuracy calculation 
                acc = (logits.argmax(1) == labels).float().mean()
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{acc.item():.2f}'})
            else:
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        val_loss, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg)
        scheduler.step()

        # Log Epoc
        wandb.log({
            "train/loss": total_loss/len(train_loader), 
            "train/acc": avg_acc,
            "val/loss": val_loss,
            "val/acc": val_acc, 
            "epoch": epoch+1,
            "lr": optimizer.param_groups[0]['lr'] 
        })

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, ckpt_dir / "last_model.pth")


def run_testing(cfg, device, model, ckpt_dir):
    print("--- FASE TESTING ---")
    checkpoint = torch.load(ckpt_dir / "last_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loader = prepare_loader(cfg, split='test')
    
    # During test use a simply classifier 
    classifier = nn.Linear(512, 10).to(device) 
    classifier.load_state_dict(checkpoint['classifier_state_dict']) # Carica i pesi addestrati!
    classifier.eval()

    
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs = imgs.to(device)

            labels = labels.to(device) - 1

            h = model(imgs, return_features=True)

            logits = classifier(h)
            preds = logits.argmax(1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    print(f"Accuracy finale sul Test Set: {100 * correct / total:.2f}%")
    wandb.log({"test/accuracy": correct / total})

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    wandb.init(project=cfg.logger.project, group=cfg.experiment.mode, config=OmegaConf.to_container(cfg))

    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)

    
    stage = cfg.get("stage", "all")
    
    if stage in ["all", "training"]:
        run_training(cfg, device, model, ckpt_dir)
    
    if stage in ["all", "testing"]:
        run_testing(cfg, device, model, ckpt_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
