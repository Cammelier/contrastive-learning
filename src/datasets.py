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
    ])

def prepare_loader(cfg, split: str = 'train'):
    """
    Optimized Loader: only loads raw tensors. 
    Augmentations and Normalization are handled by Kornia on GPU.
    """
    cfg_data = cfg.data
    
    # 1. Minimal Transform: just Resize and ToTensor
    # We resize here to ensure memory stability in the DataLoader
    transform = T.Compose([
        T.Resize((cfg_data.size, cfg_data.size)),
        T.ToTensor(),
    ])

    # 2. Load Dataset
    if split == 'val':
        # Create a validation set by splitting the 5000 labeled training images
        full_train = STL10(root=cfg_data.root_path, split='train', download=True, transform=transform)
        train_len = int(len(full_train) * 0.9)
        val_len = len(full_train) - train_len

        # Use a fixed seed for reproducibility of the split
        _, val_dataset = random_split(
            full_train, [train_len, val_len], 
            generator=torch.Generator().manual_seed(cfg.seed)
        )
        dataset = val_dataset
    else:
        # 'train', 'test', or 'unlabeled'
        dataset = STL10(root=cfg_data.root_path, split=split, download=True, transform=transform)

    # 3. Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size, 
        shuffle=True if split in ['train', 'unlabeled'] else False,
        num_workers=cfg_data.num_workers,
        pin_memory=False, # Set to False for CPU-based environments/VMs
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
