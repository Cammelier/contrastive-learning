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

# Inizializzazione Logger
logger = logging.getLogger(__name__)

# --- GPU/CPU TRANSFORMS ---
def get_gpu_transforms(device):
    mean = torch.tensor([0.4467, 0.4398, 0.4066])
    std = torch.tensor([0.2603, 0.2566, 0.2713])
    
    # Augmentations per il training
    train_aug = nn.Sequential(
        K.RandomResizedCrop(size=(96, 96), scale=(0.2, 1.0)),
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8),
        K.RandomGrayscale(p=0.2),
        K.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0), p=0.5),
        K.Normalize(mean=mean, std=std)
    ).to(device)

    # Trasformazioni per validation/test (solo normalizzazione)
    val_aug = nn.Sequential(
        K.Resize(size=(96, 96)), 
        K.CenterCrop(size=(96, 96)),
        K.Normalize(mean=mean, std=std)
    ).to(device)

    return train_aug, val_aug

# --- VALIDATION LOOP ---
def run_validation(model, classifier, val_loader, criterion, device, cfg, val_transform):
    model.eval()
    classifier.eval()
    val_loss, val_correct, val_samples = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long() 
            
            x_i = val_transform(imgs)
            
            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                h_i, z_i = model(x_i)
                loss = criterion(z_i, labels if cfg.experiment.supervised else None)
                val_loss += loss.item()

                if labels.min() >= 0:
                    logits = classifier(h_i.detach()) 
                    preds = logits.argmax(1)
                    val_correct += (preds == labels).sum().item()
                    val_samples += labels.size(0)

    avg_loss = val_loss / len(val_loader)
    avg_acc = (val_correct / val_samples) if val_samples > 0 else 0
    return avg_loss, avg_acc


def run_training(cfg, device, model, ckpt_dir):
    # --- 1. PREPARAZIONE DATI E AUGMENTATION ---
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.supervised else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')
    gpu_aug, gpu_val_aug = get_gpu_transforms(device)

    target_bs = 1024
    accumulation_steps = max(1, target_bs // cfg.batch_size)
    
    # --- 2. Weight decay filter
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    base_optimizer = torch.optim.SGD([
        {'params': decay_params, 'weight_decay': cfg.experiment.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=cfg.experiment.learning_rate, momentum=0.9)

    optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)

    # --- 3. SCHEDULER CON LINEAR WARMUP ---
    warmup_epochs = 10
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.experiment.epochs - warmup_epochs, eta_min=0
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    # --- 4. ONLINE LINEAR PROBING ---
    feat_dim = cfg.model_config.hidden_dim if hasattr(cfg.model_config, 'hidden_dim') else 512
    classifier = nn.Linear(feat_dim, 10).to(device)
    cls_optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9) # LR leggermente più alto

    # Scaler per Mixed Precision
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    criterion = ContrastiveLoss(temperature=cfg.experiment.temperature, supervised=cfg.experiment.supervised).to(device)

    # --- 5. TRAINING LOOP ---
    for epoch in range(cfg.experiment.epochs):
        model.train()
        classifier.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.experiment.epochs}")
        
        optimizer.zero_grad()

        for i, (imgs, labels) in enumerate(pbar):
            if isinstance(imgs, list): imgs = imgs[0]
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long() 

            # Augmentation su GPU
            with torch.no_grad():
                x_i = gpu_aug(imgs)
                x_j = gpu_aug(imgs)
            
            # Forward Encoder
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                x_combined = torch.cat([x_i, x_j], dim=0)
                h_combined, z_combined = model(x_combined)
                loss = criterion(z_combined, labels if cfg.experiment.supervised else None)
                scaled_loss = loss / accumulation_steps
            
            # Backward Encoder
            scaler.scale(scaled_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                optimizer.zero_grad()

            # --- Online Linear Probing ---
            if labels.min() >= 0:
                cls_optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    h_i = h_combined[:imgs.size(0)].detach() 
                    logits = classifier(h_i)
                    cls_loss = F.cross_entropy(logits, labels)

                scaler.scale(cls_loss).backward()
                scaler.step(cls_optimizer)
                
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            # Update finale dello scaler (una volta per batch o step di accumulo)
            if (i + 1) % accumulation_steps == 0:
                scaler.update()

            total_loss += loss.item()
            metrics = {'loss': f'{loss.item():.3f}'}
            if total_samples > 0: metrics['acc'] = f'{(total_correct / total_samples):.2%}'
            pbar.set_postfix(metrics)
        
        # Validazione
        val_loss_avg, val_acc = run_validation(model, classifier, val_loader, criterion, device, cfg, gpu_val_aug)
        
        wandb.log({
            "train/loss": total_loss / len(train_loader), 
            "train/acc": (total_correct / total_samples) if total_samples > 0 else 0.0,
            "val/loss": val_loss_avg,
            "val/acc": val_acc, 
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr'] 
        })
        
        scheduler.step()

    # Salvataggio
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, ckpt_dir / "last_model.pth")


# --- TESTING LOOP ---
def run_testing(cfg, device, model, ckpt_dir):
    print("--- TESTING PHASE ---")
    
    checkpoint = torch.load(ckpt_dir / "last_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loader = prepare_loader(cfg, split='test')
    _, gpu_test_aug = get_gpu_transforms(device)
    
    feat_dim = cfg.model_config.hidden_dim if hasattr(cfg.model_config, 'hidden_dim') else 512
    classifier = nn.Linear(feat_dim, 10).to(device) 
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier.eval()

    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) 

            imgs = gpu_test_aug(imgs)

            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                h = model(imgs, return_features=True)
                logits = classifier(h)
                
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    final_acc = 100 * correct / total
    print(f"Final Test Set Accuracy: {final_acc:.2f}%")
    wandb.log({"test/accuracy": final_acc / 100})

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
