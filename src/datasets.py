
import torch 
import torchvision.transforms as T 
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split

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

def get_standard_transform(size: int):
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

def prepare_loader(cfg, split: str = 'train'):
    
    cfg_data = cfg.data
    
    # 1. Choose transform 
    if split in ['train', 'unlabeled']:
        if cfg.experiment.mode == 'supervised':

            transform = T.Compose([
                T.RandomResizedCrop(cfg_data.size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            ])
        else:
            
            transform = ContrastiveTransformations(
                size=cfg_data.size,
                s=cfg_data.augmentation.color_jitter_strength
            )
    else:

        transform = get_standard_transform(cfg_data.size)

    # 2. Load Dataset and split Validation
    if split == 'val':
        # Create a val dataset from train 
        full_train = STL10(root=cfg_data.root_path, split='train', download=True, transform=transform)
        train_len = int(len(full_train) * 0.9)
        val_len = len(full_train) - train_len

        _, val_dataset = random_split(full_train, [train_len, val_len], 
                                     generator=torch.Generator().manual_seed(cfg.seed))
        dataset = val_dataset
    else:
        dataset = STL10(root=cfg_data.root_path, split=split, download=True, transform=transform)

    # 3. Creazione DataLoader
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size if split != 'test' else cfg.batch_size, 
        shuffle=True if split in ['train', 'unlabeled'] else False,
        num_workers=cfg_data.num_workers,
        pin_memory=cfg_data.augmentation.pin_memory,
        drop_last=True if split in ['train', 'unlabeled'] else False
    )

    return loader

def prepare_all_loaders(cfg):
    """Restituisce i tre loader principali (come nello schema del dottorando)."""
    train_loader = prepare_loader(cfg, split='train' if cfg.experiment.mode == 'supervised' else 'unlabeled')
    val_loader = prepare_loader(cfg, split='val')
    test_loader = prepare_loader(cfg, split='test')
    return train_loader, val_loader, test_loader
