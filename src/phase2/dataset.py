import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "phase1"))

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import AUGMENTED_DIR, CLASSES, IMAGE_SIZE


class NailDataset(Dataset):
    def __init__(self, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(CLASSES)}

        for class_name in CLASSES:
            class_dir = AUGMENTED_DIR / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[class_name]))

        print(f"[DATASET] Loaded {len(self.samples)} samples across {len(CLASSES)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_dataloaders(batch_size=32):
    train_tf, val_tf = get_transforms()
    full_dataset = NailDataset(transform=train_tf)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    import torch
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = val_tf

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"[DATALOADER] Train: {train_size} | Val: {val_size}")
    return train_loader, val_loader