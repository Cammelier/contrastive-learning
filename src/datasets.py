import torch 
import torchvision.transforms as T 
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split

class ContrastiveTransformations:
    """
    Generates two different random augmentations of the same image.
    This is essential for Contrastive Learning (SimCLR and SupCon) 
    to prevent the model from finding trivial solutions.
    """
    def __init__(self, size: int, s: float = 0.5):
        # Color jitter strength adjusted by 's'
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __call__(self, x):
        # Returns a list of two differently augmented views of image x
        return [self.transform(x), self.transform(x)]

def get_standard_transform(size: int):
    """
    Standard transformations for validation and testing.
    No random augmentations are applied here.
    """
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

def prepare_loader(cfg, split: str = 'train'):
    """
    Prepares the DataLoader for specific splits of STL10.
    """
    cfg_data = cfg.data
    
    # 1. Choose transformation logic
    if split in ['train', 'unlabeled']:
        # If we are doing any form of Contrastive Learning (Self-Supervised or Supervised),
        # we NEED the double view to avoid loss collapse (diagonal = 1.0)
        if cfg.experiment.mode in ['self_supervised', 'supervised']:
            transform = T.Compose([
        T.ToTensor(), # Carica solo l'immagine base
            ])
        else:
            # Standard supervised case (e.g., standard Cross-Entropy)
            transform = T.Compose([
                T.RandomResizedCrop(cfg_data.size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            ])
    else:
        # Validation and Test sets use fixed standard transforms
        transform = get_standard_transform(cfg_data.size)

    # 2. Load Dataset and handle Validation split
    if split == 'val':
        # Create a validation set by splitting the 5000 labeled training images
        full_train = STL10(root=cfg_data.root_path, split='train', download=True, transform=transform)
        train_len = int(len(full_train) * 0.9)
        val_len = len(full_train) - train_len

        generator = torch.Generator().manual_seed(cfg.seed)

        # Use a fixed seed for reproducibility of the split
        _, val_dataset = random_split(full_train, [train_len, val_len], 
                                     generator=torch.Generator().manual_seed(cfg.seed))
        dataset = val_dataset
    else:
        # Standard STL10 splits: 'train' (5k), 'test' (8k), or 'unlabeled' (100k)
        dataset = STL10(root=cfg_data.root_path, split=split, download=True, transform=transform)

    # 3. Create DataLoader
    # Note: drop_last is True for training to maintain stable batch statistics in Contrastive Loss
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size, 
        shuffle=True if split in ['train', 'unlabeled'] else False,
        num_workers=cfg_data.num_workers,
        pin_memory=cfg_data.augmentation.pin_memory,
        drop_last=True if split in ['train', 'unlabeled'] else False
    )

    return loader

def prepare_all_loaders(cfg):
    """
    Returns the three main loaders: Train, Val, and Test.
    """
    # Self-supervised uses 'unlabeled' for training, Supervised uses 'train'
    train_split = 'train' if cfg.experiment.mode == 'supervised' else 'unlabeled'
    
    train_loader = prepare_loader(cfg, split=train_split)
    val_loader = prepare_loader(cfg, split='val')
    test_loader = prepare_loader(cfg, split='test')
    
    return train_loader, val_loader, test_loader
