import hydra 
import torch 
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path 
from src.datasets import get_stl10_dataloader
from src.torch.models import SimCLR
from src.losses import ContrastiveLoss
from src.torch.engine import ContrastiveEngine

@hydra.main(version_base=None,config_path="config", config_name="configuratore")
def main(cfg: DictConfig):
    # 1. Setup path and environment
    root_dir = Path(hydra.utils.get_original_cwd())
    ckpt_dir = root_dir/"checkpoints"/cfg.experiment.mode
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
    loader = get_stl10_dataloader(cfg)

    # 4. Model, criterion, optimizer
    model = SimCLR(cfg).to(device)

    criterion = ContrastiveLoss(
        temperature=cfg.experiment.temperature,
        supervised = cfg.experiment.supervised
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-4
    )

    # 5. Training loop
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0

        for batch_idx, (imgs,labels) in enumerate(loader):
            x_i, x_j = imgs[0].to(device), imgs[1].to(device)
            labels = labels.to(device)

        # Forward pass
        _, z_i = model(x_i)
        _, z_j = model(x_j)

        features = torch.stack([z_i, z_j], dim=0) 

        # Calculate loss 
        loss = criterion(features, labels)

        # Backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            wandb.log({"Batch Loss": loss.item()})

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{cfg.epochs}] | Avg Loss: {avg_loss:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "Avg_epoch_loss": avg_loss,
        "learning_rate": optimizer.param_groups[0]['lr']
        })
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0 or (epoch+1) == cfg.epochs:
        ckpt_path = ckpt_dir/f"model_epoch_{epoch+1}.pth"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            "loss": avg_loss,
        },ckpt_path)
    
    wandb.finish()
    print("--- Esperimento Terminato ---")
    if __name__ == "__main__":
        main()
