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
        K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                    std=torch.tensor([0.247, 0.243, 0.261]))
    ).to(device)

    val_aug = nn.Sequential(
        K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]), 
                    std=torch.tensor([0.247, 0.243, 0.261]))
        ).to(device)

    return train_aug, val_aug

def run_validation(model, classifier, val_loader, criterion, device, cfg, val_transform):
    """
    Performs validation during training to monitor Contrastive Loss and Online Accuracy.
    OPTIMIZED: Uses GPU Normalization and Mixed Precision.
    """
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            # --- INPUT HANDLING ---
            # Move raw images to GPU immediately. 
            # We assume the DataLoader now returns a single tensor (no list), 
            # because we removed the CPU augmentations.
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # --- GPU TRANSFORM ---
            # Apply deterministic normalization on GPU (via Kornia)
            # This is critical because the raw images from DataLoader are not normalized yet.
            x_i = val_transform(imgs)
            x_j = x_i # For validation, we use the same view (duplicate) to compute loss
            
            # --- MIXED PRECISION CONTEXT ---
            # Even in validation, Autocast saves memory and speeds up inference on T4
            with torch.cuda.amp.autocast():
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
    OPTIMIZED: Uses GPU Augmentations (Kornia) and Mixed Precision (AMP).
    """
    # 1. Prepare Data Loaders
    # IMPORTANT: Ensure your DataLoader returns ONLY ToTensor() images (no heavy augs on CPU)
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.supervised else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')

    # --- GPU AUGMENTATIONS SETUP ---
    # Retrieve the augmentation pipelines defined earlier
    gpu_aug, gpu_val_aug = get_gpu_transforms(device)

    # DYNAMIC ACCUMULATION STEPS
    if cfg.experiment.mode == "self_supervised":
        target_bs = 256
    else:
        target_bs = 128
    
    accumulation_steps = max(1, target_bs // cfg.batch_size)
    
    logger.info(f"Mode: {cfg.experiment.mode}")
    logger.info(f"Accumulation Steps: {accumulation_steps}") 
    
    # 2. Criterion, Optimizer & Scheduler
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised=cfg.experiment.supervised
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.experiment.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.experiment.epochs)

    # --- MIXED PRECISION SCALER ---
    # Essential for Tesla T4 to run efficiently in FP16
    scaler = torch.cuda.amp.GradScaler()

    # 3. Online Linear Classifier (trained on top of frozen features)
    # This monitors representation quality during training
    classifier = nn.Linear(cfg.model_config.hidden_dim if hasattr(cfg.model_config, 'out_dim') else 512, 10).to(device)
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3) 

    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0

        optimizer.zero_grad()
        cls_optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        for i, (imgs, labels) in enumerate(pbar):
            
            # --- INPUT HANDLING & GPU TRANSFER ---
            # Move raw images to GPU immediately. 
            # non_blocking=True allows async transfer while GPU is busy.
            if isinstance(imgs, list):
                imgs = imgs[0]

            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --- GPU AUGMENTATION ---
            # Generate two views directly on GPU using Kornia.
            # This completely removes the CPU bottleneck.
            with torch.no_grad():
                x_i = gpu_aug(imgs)
                x_j = gpu_aug(imgs)
            
            
            # --- STEP 1: Contrastive Learning (Backbone update) ---
            # Enable Mixed Precision for the forward pass
            with torch.cuda.amp.autocast():
                x_combined = torch.cat([x_i, x_j], dim=0)
                h_combined, z_combined = model(x_combined)
                z_i, z_j = torch.split(z_combined, x_i.size(0))
                
                # Calculate Contrastive Loss
                # If supervised=True, uses labels for SupCon Loss
                loss = criterion(torch.cat([z_i, z_j], dim=0), labels if cfg.experiment.supervised else None)
                
                # Normalize loss for gradient accumulation
                scaled_loss = loss / accumulation_steps
            
            # Backward pass with Scaler (handles FP16 gradients)
            scaler.scale(scaled_loss).backward()
           
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

            # --- STEP 2: Online Linear Evaluation (Classifier update) ---
            # We calculate accuracy only on valid labels
            
            has_valid_labels = (labels.min() >= 0)
            current_acc = 0.0

            if has_valid_labels:
                # Use Autocast here as well for speed
                with torch.cuda.amp.autocast():
                    # Detach features! We train ONLY the classifier here.
                    # We don't want classifier gradients affecting the backbone.
                    h_i = h_combined[:x_i.size(0)].detach() 
                    
                    logits = classifier(h_i)
                    cls_loss = F.cross_entropy(logits, labels)
                
                cls_optimizer.zero_grad()
                # Scale the classifier loss too
                scaler.scale(cls_loss).backward()
                scaler.step(cls_optimizer)
                scaler.update()

                # Calculate Accuracy
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                total_correct += correct
                total_samples += labels.size(0)
                
                current_acc = total_correct / total_samples

            # --- UPDATE PROGRESS BAR ---
            postfix_dict = {'loss': f'{loss.item():.3f}'}
            
            if total_samples > 0:
                postfix_dict['acc'] = f'{current_acc:.2%}'
            
            pbar.set_postfix(postfix_dict)
        
        # End of Epoch: Validation
        # Pass the GPU validation transform to the validation function
        val_loss, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg, gpu_val_aug)
        
        scheduler.step()

        # WandB logging
        wandb.log({
            "train/loss": total_loss / len(train_loader), 
            "train/acc": (total_correct / total_samples) if total_samples > 0 else 0.0,
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
            with torch.cuda.amp.autocast():
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

    #if torch.cuda.device_count() > 1:
     #   logger.info(f"🔥 ATTIVAZIONE MULTI-GPU: Trovate {torch.cuda.device_count()} GPU!")
      #  model = nn.DataParallel(model)
        
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
