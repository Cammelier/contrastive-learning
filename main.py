import hydra 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import wandb
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 

# Custom imports (ensure these modules exist in your src folder)
from src.datasets import prepare_loader
from src.models import SimCLR
from src.losses import ContrastiveLoss

# Initialize Logger
logger = logging.getLogger(__name__)

def run_validation(model, classifier, val_loader, criterion, device, cfg):
    """
    Performs validation during training to monitor Contrastive Loss and Online Accuracy.
    """
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            # Handle input: STL-10 might return a list of 2 views
            if isinstance(imgs, list):
                x_i, x_j = imgs[0].to(device), imgs[1].to(device)
            else:
                x_i = imgs.to(device)
                x_j = x_i
            
            labels = labels.to(device)
            if cfg.experiment.supervised:
                labels = labels - 1 # Adjust 1-based STL-10 labels to 0-based
            
            # 1. Contrastive Validation Loss 
            x_combined = torch.cat([x_i, x_j], dim=0)
            h_combined, z_combined = model(x_combined)
            z_i, z_j = torch.split(z_combined, x_i.size(0))
            
            loss = criterion(torch.cat([z_i, z_j], dim=0), labels if cfg.experiment.supervised else None)
            val_loss += loss.item()

            # 2. Online Linear Probing Accuracy
            if labels.min() >= 0:
                h_i = h_combined[:x_i.size(0)]
                logits = classifier(h_i) 
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    avg_acc = (val_correct / val_samples) if val_samples > 0 else 0
    return avg_loss, avg_acc

def run_training(cfg, device, model, ckpt_dir):
    """
    Main training loop with dynamic Gradient Accumulation and Online Linear Probing.
    """
    # 1. Prepare Data Loaders
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.supervised else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')

    # DYNAMIC ACCUMULATION STEPS: 
    # Balance GPU memory (6GB) while maintaining large batches for Contrastive Learning.
    if cfg.experiment.mode == "self_supervised":
        target_bs = 256  # SimCLR benefits from very large batches
    else:
        target_bs = 128  # Supervised Contrastive (SupCon) works well with medium batches
    
    # Calculate steps based on physical batch_size provided in cfg
    accumulation_steps = max(1, target_bs // cfg.batch_size)
    
    logger.info(f"Mode: {cfg.experiment.mode}")
    logger.info(f"Physical Batch: {cfg.batch_size}, Virtual Target: {target_bs}")
    logger.info(f"Gradient Accumulation Steps: {accumulation_steps}") 
    
    # 2. Criterion, Optimizer & Scheduler
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=cfg.experiment.supervised
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.experiment.epochs)

    # 3. Online Linear Classifier (trained on top of frozen features)
    classifier = nn.Linear(512, 10).to(device)
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.experiment.learning_rate)

    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        for i, (imgs, labels) in enumerate(pbar):
            
            if isinstance(imgs, list):
                x_i, x_j = imgs[0].to(device), imgs[1].to(device)
            else: 
                x_i = imgs.to(device)
                x_j = x_i 
            
            labels = labels.to(device)
            if cfg.experiment.supervised:
                labels = labels - 1 

            # --- STEP 1: Contrastive Learning (Backbone update) ---
            x_combined = torch.cat([x_i, x_j], dim=0)
            h_combined, z_combined = model(x_combined)
            z_i, z_j = torch.split(z_combined, x_i.size(0))
            
            # Loss calculation
            loss = criterion(torch.cat([z_i, z_j], dim=0), labels if cfg.experiment.supervised else None)
            
            # Scaled loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
           
            # Optimizer step every 'accumulation_steps' iterations
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            # --- STEP 2: Online Linear Evaluation (Classifier update) ---
            if labels.min() >= 0:
                # Detach features to ensure backbone weights are not updated here
                h_i = h_combined[:x_i.size(0)].detach() 
                logits = classifier(h_i)
                cls_loss = F.cross_entropy(logits, labels)
                
                cls_optimizer.zero_grad()
                cls_loss.backward()
                cls_optimizer.step()

                # Accuracy metrics
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
                
                current_acc = total_correct / total_samples
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{current_acc:.2f}'})
            else:
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        # Epoch-level validation and logging
        avg_train_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
        val_loss, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg)
        
        scheduler.step()

        # WandB logging
        wandb.log({
            "train/loss": total_loss / len(train_loader), 
            "train/acc": avg_train_acc,
            "val/loss": val_loss,
            "val/acc": val_acc, 
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr'] 
        })

    # Final Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, ckpt_dir / "last_model.pth")


def run_testing(cfg, device, model, ckpt_dir):
    """
    Evaluation on the official Test Set using the trained backbone and linear head.
    """
    print("--- TESTING PHASE ---")
    checkpoint = torch.load(ckpt_dir / "last_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loader = prepare_loader(cfg, split='test')
    
    classifier = nn.Linear(512, 10).to(device) 
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            imgs = imgs.to(device)
            labels = labels.to(device) - 1

            # Extract features only
            h = model(imgs, return_features=True)

            logits = classifier(h)
            preds = logits.argmax(1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    final_acc = 100 * correct / total
    print(f"Final Test Set Accuracy: {final_acc:.2f}%")
    wandb.log({"test/accuracy": final_acc / 100})

@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # Setup directories and device
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # Init WandB
    wandb.init(
        project=cfg.logger.project, 
        group=cfg.experiment.mode, 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Build Model
    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)

    # Execution Stages
    stage = cfg.get("stage", "all")
    
    if stage in ["all", "training"]:
        run_training(cfg, device, model, ckpt_dir)
    
    if stage in ["all", "testing"]:
        run_testing(cfg, device, model, ckpt_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
