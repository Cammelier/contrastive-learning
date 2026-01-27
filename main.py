import hydra 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import wandb
import kornia.augmentation as K
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

def get_gpu_transforms(device):
    # Training Augmentations (Random)
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.2, 1.0)),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), p=0.5),
        K.Normalize(mean=torch.tensor([0.4467, 0.4398, 0.4066]), 
                    std=torch.tensor([0.2603, 0.2566, 0.2713]))
    ).to(device)

    val_aug = nn.Sequential(
    K.Resize(size=(96, 96)), 
    K.CenterCrop(size=(96, 96)),
    K.Normalize(
        mean=torch.tensor([0.4467, 0.4398, 0.4066], device=device), 
        std=torch.tensor([0.2603, 0.2566, 0.2713], device=device)
    )
).to(device)


    return train_aug, val_aug

def run_validation(model, classifier, val_loader, criterion, device, cfg, val_transform):
    """
    Validation loop: computes Contrastive Loss and Online Linear Probing Accuracy.
    """
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            # STL-10 labels are 1-indexed (1-10). Shift to 0-9 for CrossEntropy and indexing.
            labels = labels.to(device, non_blocking=True).long() 
            
            # Apply GPU normalization
            x_i = val_transform(imgs)
            
            with torch.amp.autocast(device_type=device.type):
                # Forward pass
                h_i, z_i = model(x_i)
                
                # Validation Loss: using a single view for efficiency
                # ContrastiveLoss handles single view by comparing against itself (if implemented)
                # or we pass it directly to monitor convergence.
                loss = criterion(z_i, labels if cfg.experiment.supervised else None)
                val_loss += loss.item()

                # Calculate Accuracy (Online Linear Probing)
                if labels.min() >= 0:
                    logits = classifier(h_i.detach()) 
                    preds = logits.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    avg_acc = (val_correct / val_samples) if val_samples > 0 else 0
    return avg_loss, avg_acc


def run_training(cfg, device, model, ckpt_dir):
    """
    Optimized training loop for STL-10 with GPU Augmentations and Mixed Precision.
    """
    # 1. Setup Data Loaders
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.supervised else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')
    gpu_aug, gpu_val_aug = get_gpu_transforms(device)

    # 2. Training Dynamics
    target_bs = 1024
    accumulation_steps = max(1, target_bs // cfg.batch_size)
    
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=cfg.experiment.supervised
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.experiment.epochs)
    scaler = torch.amp.GradScaler()

    # 3. Online Linear Classifier (trained on frozen features)
    feat_dim = cfg.model_config.hidden_dim if hasattr(cfg.model_config, 'hidden_dim') else 512
    classifier = nn.Linear(feat_dim, 10).to(device)
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3) 

    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        optimizer.zero_grad()
        cls_optimizer.zero_grad()

        for i, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long() 

            # GPU-based augmentations
            with torch.no_grad():
                x_i = gpu_aug(imgs)
                x_j = gpu_aug(imgs)
            
            # --- BACKBONE UPDATE ---
            with torch.amp.autocast(device_type=device.type):
                x_combined = torch.cat([x_i, x_j], dim=0)
                h_combined, z_combined = model(x_combined)
                loss = criterion(z_combined, labels if cfg.experiment.supervised else None)
                scaled_loss = loss / accumulation_steps
            
            scaler.scale(scaled_loss).backward()
           
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # --- CLASSIFIER UPDATE ---
            if labels.min() >= 0:
                with torch.amp.autocast(device_type=device.type):
                    h_i = h_combined[:imgs.size(0)].detach() 
                    logits = classifier(h_i)
                    cls_loss = F.cross_entropy(logits, labels)

            

                cls_optimizer.zero_grad()
                scaler.scale(cls_loss).backward()
                scaler.step(cls_optimizer)
                scaler.update()

                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                
            # Update progress bar safely
            metrics = {'loss': f'{loss.item():.3f}'}
            if total_samples > 0:
                acc_value = total_correct / total_samples
                metrics['acc'] = f'{acc_value:.2%}'
            else:
                metrics['acc'] = '0.00%' # Default if no valid labels yet
            pbar.set_postfix(metrics)
        
        # Validation Phase
        val_loss_avg, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg, gpu_val_aug)
        print(f"Epoch {epoch+1} Summary: Val Loss {val_loss_avg:.4f}, Val Acc {val_acc:.2%}")
        
        # --- FIXED WANDB LOGGING (Inside Epoch Loop) ---
        wandb.log({
            "train/loss": total_loss / len(train_loader), 
            "train/acc": (total_correct / total_samples) if total_samples > 0 else 0.0,
            "val/loss": val_loss_avg,
            "val/acc": val_acc, 
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr'] 
        })
        
        scheduler.step()

    # Final Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, ckpt_dir / "last_model.pth")



def run_testing(cfg, device, model, ckpt_dir):
    """
    Evaluation on the official Test Set using the trained backbone and linear head.
    OPTIMIZED: Uses GPU Normalization and Mixed Precision for consistent performance.
    """
    print("--- TESTING PHASE ---")
    
    # Load best model weights
    checkpoint = torch.load(ckpt_dir / "last_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare Loader (Standard, returns raw images)
    test_loader = prepare_loader(cfg, split='test')
    
    # --- GPU TRANSFORM SETUP ---
    # We retrieve the validation transform (which only does Normalization)
    # because the test loader sends raw tensors just like the training loader now.
    _, gpu_test_aug = get_gpu_transforms(device)
    
    # Rebuild Classifier
    # Ideally, dimensions should come from config, but we stick to 512 as per your snippet
    classifier = nn.Linear(512, 10).to(device) 
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.eval()

    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            # Move to GPU immediately
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) 

            # --- GPU NORMALIZATION ---
            # Critical: Normalize the raw images on the GPU
            imgs = gpu_test_aug(imgs)

            # --- MIXED PRECISION INFERENCE ---
            # Faster inference on T4
            with torch.amp.autocast(device_type=device.type):
                # Extract features only
                h = model(imgs, return_features=True)
                logits = classifier(h)
                
            # Get predictions
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

    # --- PERFORMANCE OPTIMIZATION (CRITICAL FOR T4) ---
    # Since your input size is fixed (96x96), this enables the CuDNN autotuner.
    # It finds the most efficient convolution algorithm for your specific hardware.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # Init WandB
    wandb.init(
        project=cfg.logger.project, 
        group=cfg.experiment.mode, 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # Build Model
    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)

    # if torch.cuda.device_count() > 1:
    #    logger.info(f"🔥 ATTIVAZIONE MULTI-GPU: Trovate {torch.cuda.device_count()} GPU!")
    #    model = nn.DataParallel(model)
        
    # Execution Stages
    stage = cfg.get("stage", "all")
    
    if stage in ["all", "training"]:
        run_training(cfg, device, model, ckpt_dir)
    
    if stage in ["all", "testing"]:
        # run_testing handles loading the best checkpoint internally
        run_testing(cfg, device, model, ckpt_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
