import hydra 
import torch 
from torchlars import LARS
import torch.nn as nn
import torch.nn.functional as F
import wandb
import kornia.augmentation as K
import logging
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from tqdm import tqdm 

# Assicurati che questi moduli esistano nella tua cartella src
from src.datasets import prepare_loader
from src.models import SimCLR
from src.losses import ContrastiveLoss

import warnings
# Zittisce il warning specifico di torchlars
warnings.filterwarnings("ignore", message="This overload of add_ is deprecated")


# --- GPU/CPU TRANSFORMS ---
def get_gpu_transforms(device):
    mean = torch.tensor([0.4467, 0.4398, 0.4066])
    std = torch.tensor([0.2603, 0.2566, 0.2713])
    
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.08, 1.0)), 
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), p=0.5),
        K.Normalize(mean=mean, std=std)
    ).to(device)


    # Validation/Test transformations: standard Resize + CenterCrop + Normalize
    val_aug = nn.Sequential(
        K.Resize(size=(96, 96)), 
        K.CenterCrop(size=(96, 96)),
        K.Normalize(mean=mean, std=std)
    ).to(device)

    return train_aug, val_aug

# --- VALIDATION LOOP ---
def run_validation(model, classifier, val_loader, criterion, device, cfg, val_transform):
    """
    Evaluates both the contrastive loss and the online linear probe performance.
    Optimized for RTX 4090 using BFloat16.
    """
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    # Use BFloat16 if supported 
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            # Handle list of images (multi-view) if returned by the loader
            if isinstance(imgs, list): imgs = imgs[0]
            
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long() 
            
            # Fast GPU-side normalization
            x_val = val_transform(imgs)
            
            # Inference using mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                h_i, z_i = model(x_val)
                
                # Contrastive Loss (Supervised or Self-Supervised)
                loss = criterion(z_i, labels if cfg.experiment.supervised else None)
                val_loss += loss.item()

                # Accuracy tracking for Online Linear Probing
                if labels.min() >= 0:
                    # Detach features to ensure evaluation doesn't track gradients
                    logits = classifier(h_i.detach()) 
                    preds = logits.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    avg_acc = (val_correct / val_samples) if val_samples > 0 else 0
    
    return avg_loss, avg_acc



def run_training(cfg, device, model, ckpt_dir):
    
    # Enable Tensor Cores for Float32 matrix multiplications (Significant speedup)
    torch.set_float32_matmul_precision('high')

    # --- 1. DATA PREPARATION & AUGMENTATION ---
   
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.supervised else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')
    gpu_aug, gpu_val_aug = get_gpu_transforms(device)

    # Gradient accumulation setup to simulate larger batch sizes (Target: 1024)
    target_bs = 1024
    accumulation_steps = max(1, target_bs // cfg.batch_size)
    
    # --- 2. PARAMETER FILTERING (WEIGHT DECAY) ---
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(nd in name for nd in ["bias", "bn", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Base optimizer for LARS wrapper
    base_optimizer = torch.optim.SGD([
        {'params': decay_params, 'weight_decay': cfg.experiment.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=cfg.experiment.learning_rate, momentum=0.9)

    # LARS: Crucial for large batch contrastive learning stability
    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    # --- 3. SCHEDULER (LINEAR WARMUP + COSINE ANNEALING) ---
    warmup_epochs = 10
    total_epochs = cfg.experiment.epochs
    
    # Linear warmup prevents early divergence with high learning rates
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, total_iters=warmup_epochs
    )
    # Cosine decay follows the original SimCLR/MoCo recipe
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=0
    )
    # SequentialLR handles the transition between warmup and decay
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # --- 4. ONLINE LINEAR PROBING ---
    # Tracks representation quality during training without blocking gradients (using .detach())
    feat_dim = cfg.model_config.hidden_dim if hasattr(cfg.model_config, 'hidden_dim') else 512
    # Sequential head with BN (affine=False) provides a more stable probe
    classifier = nn.Sequential(
        nn.BatchNorm1d(feat_dim, affine=False),
        nn.Linear(feat_dim, 10)
    ).to(device)
    cls_optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9)

    # Mixed Precision Setup: BFloat16 is natively supported and faster on RTX 4090
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature, 
        supervised=cfg.experiment.supervised
    ).to(device)

    # --- 5. MAIN TRAINING LOOP ---
    for epoch in range(total_epochs):
        model.train()
        classifier.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        
        optimizer.zero_grad()

        for i, (imgs, labels) in enumerate(pbar):
            # Compatibility check for multi-view datasets
            if isinstance(imgs, list): imgs = imgs[0]
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True).long()

            # GPU-based Data Augmentation
            with torch.no_grad():
                x_i, x_j = gpu_aug(imgs), gpu_aug(imgs)
            
            # --- FORWARD ENCODER (Autocast BF16/FP16) ---
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                # Process concatenated views (SimCLR style)
                h_combined, z_combined = model(torch.cat([x_i, x_j], dim=0))
                loss = criterion(z_combined, labels if cfg.experiment.supervised else None)
                scaled_loss = loss / accumulation_steps
            
            # Scaled Backward
            scaler.scale(scaled_loss).backward()
            
            # Optimizer Step with Gradient Accumulation
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()

            # --- ONLINE LINEAR PROBING STEP ---
            if labels.min() >= 0: # Only run if valid labels exist
                cls_optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    # Detach h to ensure probing doesn't affect encoder learning
                    h_i = h_combined[:imgs.size(0)].detach() 
                    logits = classifier(h_i)
                    cls_loss = F.cross_entropy(logits, labels)

                scaler.scale(cls_loss).backward()
                scaler.step(cls_optimizer)
                
                # Internal Accuracy Tracking
                total_correct += (logits.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)

            # Scaler update after all optimizer steps in the current iteration
            if (i + 1) % accumulation_steps == 0:
                scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}', 
                'acc': f'{(total_correct/total_samples if total_samples>0 else 0):.2%}'
            })
        
        # --- VALIDATION & LOGGING ---
        val_loss, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg, gpu_val_aug)
        
        wandb.log({
            "train/loss": total_loss / len(train_loader), 
            "train/acc": total_correct / total_samples if total_samples > 0 else 0.0,
            "val/loss": val_loss, 
            "val/acc": val_acc, 
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Step the scheduler after each epoch
        scheduler.step()

    # --- FINAL CHECKPOINT SAVE ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
        'epoch': total_epochs
    }, ckpt_dir / "last_model.pth")
    
    print(f"Training Complete. Model saved to {ckpt_dir}")



# --- TESTING LOOP  ---
def run_testing(cfg, device, model, ckpt_dir):
    print("\n--- STARTING TESTING PHASE ---")
    
    # 1. Load Checkpoint
    checkpoint = torch.load(ckpt_dir / "last_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. Setup Classifier
    feat_dim = cfg.model_config.hidden_dim if hasattr(cfg.model_config, 'hidden_dim') else 512
    classifier = nn.Linear(feat_dim, cfg.data.num_classes).to(device) 
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    model.eval()
    classifier.eval()
    
    # 3. Data & Augmentation
    test_loader = prepare_loader(cfg, split='test')
    _, gpu_test_aug = get_gpu_transforms(device)
    
    
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    correct, total = 0, 0
    all_preds, all_labels = [], []

    # 4. Evaluation Loop
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            if isinstance(imgs, list): imgs = imgs[0]
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long() 

            # Augmentation di validazione (es. normalization)
            imgs = gpu_test_aug(imgs)

            # Sfrutta i Tensor Cores della 4090 con BF16
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                # Assicurati che return_features=True restituisca l'output corretto
                h = model(imgs, return_features=True)
                logits = classifier(h)
                
            preds = logits.argmax(1)
            
            # Accumulo per accuratezza
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Accumulo per matrice di confusione
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 5. Metrics & Logging
    final_acc = 100 * correct / total
    print(f"\n[TEST RESULT] Accuracy: {final_acc:.2f}%")
    
    # Log su WandB
    metrics = {"test/accuracy": final_acc / 100}
    
    # Generazione automatica Matrice di Confusione su WandB
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else None
    metrics["test/confusion_matrix"] = wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=class_names
    )
    
    wandb.log(metrics)
    return final_acc


# --- MAIN EXECUTION ---
@hydra.main(version_base="1.2", config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir / "checkpoints" / cfg.experiment.mode
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Device auto-detection
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    wandb.init(
        project=cfg.logger.project, 
        group=cfg.experiment.mode, 
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    model = SimCLR(base_model=cfg.model.backbone, out_dim=cfg.model.out_dim).to(device)

    stage = cfg.get("stage", "all")
    
    if stage in ["all", "training"]:
        run_training(cfg, device, model, ckpt_dir)
    
    if stage in ["all", "testing"]:
        run_testing(cfg, device, model, ckpt_dir)

    wandb.finish()

if __name__ == "__main__":
    main()
