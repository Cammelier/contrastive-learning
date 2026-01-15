import torch  
from torch.amp import autocast
from torch.cuda.amp import  GradScaler
import wandb
from tqdm import tqdm 

class ContrastiveEngine:
    def __init__(self, model, optimizer, criterion, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Mixed precision scaler
        self.scaler = GradScaler(enable=cfg.model.use_mixed_precision)

    def train_step(self,batch):
        self.model.train()

        # imgs is a list of two tensors [batch, 3, 96, 96]
        imgs, labels = batch 

        images = torch.cat(imgs, dim=0).to(self.device,non_blocking=True )  # Concatenate along batch dimension
        labels = labels.to(self.device,non_blocking=True )

        self.optimizer.zero_grad()

        # Mixed precision use float16 to save up memory
        with autocast(enabled=self.cfg.model.use_mixed_precision):
           # Forward pass
           _, z= self.model(images)

           # Divided output in two parts
           z_i, z_j = torch.split(z, z.shape[0] // 2, dim=0)
           z_combined = torch.stack([z_i, z_j], dim=1)  # Shape: [batch_size, 2,  out_dim]

           # Calculate loss: Supervised (SupCon) or Self_supervised(SimCLR)
           if self.cfg.experiments_type == 'supervised':
                loss = self.criterion(z_combined, labels)
           else:
               loss = self.criterion(z_combined)
        # Backward with scaler to avoid underflow in float16
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()
    
    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()

        imgs, labels = batch 
        imgs = torch.cat(imgs, dim=0).to(self.device,non_blocking=True )  # Concatenate along batch dimension
        labels = labels.to(self.device,non_blocking=True )

        with autocast(enabled=self.cfg.model.use_mixed_precision):
           # Forward pass
           _, z= self.model(images)

           # Divided output in two parts
           z_i, z_j = torch.split(z, z.shape[0] // 2, dim=0)
           z_combined = torch.stack([z_i, z_j], dim=1)  # Shape: [batch_size, 2,  out_dim]

           # Calculate loss: Supervised (SupCon) or Self_supervised(SimCLR)
           if self.cfg.experiments_type == 'supervised':
                loss = self.criterion(z_combined, labels)
           else:
               loss = self.criterion(z_combined)

        return loss.item()
    
    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(loader,desc=f"Epoch {epoch}", unit="batch")

        for batch in pbar:
            loss_val = self.train_step(batch)
            total_loss += loss_val

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})

            # Log to wandb
            wandb.log({'train/batch_loss': loss_val})
        
        avg_loss = total_loss / len(loader)
        return avg_loss