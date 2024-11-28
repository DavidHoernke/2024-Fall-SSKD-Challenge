import torch
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
from albumentations import HorizontalFlip, RandomCrop, Normalize as AlbNormalize, Compose as AlbCompose
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
from PIL import Image


class VOCSegmentationCustom(VOCSegmentation):
    def __init__(self, root, year, image_set, transforms=None, target_transforms=None):
        super().__init__(root=root, year=year, image_set=image_set, download=True)
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):
        # Load image and target from VOCSegmentation
        image, target = super().__getitem__(index)

        if self.transforms:
            image_np = np.array(image)  # Convert PIL image to numpy
            target_np = np.array(target)  # Convert PIL mask to numpy

            # Pass as named arguments
            augmented = self.transforms(image=image_np, mask=target_np)  # Pass correctly
            image, target = augmented["image"], augmented["mask"]

        if self.target_transforms:
            target = self.target_transforms(target)

        return image, target



# Preprocessing Transforms
def get_transforms(mode, fixed_size=(256, 256)):
    if mode == "train":
        return AlbCompose([
            RandomCrop(height=fixed_size[0], width=fixed_size[1], always_apply=True),
            HorizontalFlip(p=0.5),
            AlbNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    elif mode == "val":
        return AlbCompose([
            AlbNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")


# Target Mask Transform to Convert Ignore Index
def target_transform(target):
    target = np.array(target, dtype=np.int64)
    target[target == 255] = 255  # Keep the ignore index as 255
    return torch.as_tensor(target, dtype=torch.long)


# Define DataLoaders
def get_dataloaders(root="./data", batch_size=4, fixed_size=(256, 256)):
    # Train Dataset and Loader
    train_dataset = VOCSegmentationCustom(
        root=root,
        year="2012",
        image_set="train",
        transforms=get_transforms("train", fixed_size=fixed_size),
        target_transforms=target_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Validation Dataset and Loader
    val_dataset = VOCSegmentationCustom(
        root=root,
        year="2012",
        image_set="val",
        transforms=get_transforms("val", fixed_size=fixed_size),
        target_transforms=target_transform
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader


# Example Usage
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4, fixed_size=(256, 256))

    # Inspect a batch
    for images, masks in train_loader:
        print(f"Image batch shape: {images.shape}")  # Should be [B, 3, 256, 256]
        print(f"Mask batch shape: {masks.shape}")    # Should be [B, 256, 256]
        break
