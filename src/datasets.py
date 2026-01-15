import torch 
import torchvision.transforms as T 
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

class ContrastiveTransformations:
    def __init__(self, size: int, s: float = 0.5):
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        self.transform = T.Compose([
            T.RandomResizedCrop(size=size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * size) * 2 + 1, sigma=(0.1, 2.0)),
            T.ToTensor(),
           T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
    def get_stl10_dataloader(cfg, split: str = 'unlabeled'):
        cfg_data = cfg.data
        contrastive_transform = ContrastiveTransformations(
            size=cfg_data.size,
            s=cfg_data.augmentation.color_jitter_strength)
        
        dataset = STL10(
            root=cfg_data.root_path,
            split=split,
            download=True,
            transform=contrastive_transform
        )

        loader = DataLoader(
            dataset,
            batch_size=cfg_data.batch_size,
            shuffle=True if split != 'test' else False,
            num_workers=cfg_data.num_workers,
            pin_memory=cfg_data.augmentation.pin_memory,
            # drop_last avoids batches of 1 element that break contrastive loss
            drop_last=True if split != 'test' else False
        )

        return loader