import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations import Compose, Resize, HorizontalFlip, RandomCrop, Normalize, GridDistortion, GaussianBlur, ColorJitter
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


# Custom mIoU Calculation Function
def calculate_miou(pred, target, num_classes=21):
    pred = pred.cpu().numpy()  # Convert to numpy
    target = target.cpu().numpy()
    ious = []

    for cls in range(num_classes):
        if cls == 255:  # Skip ignore index
            continue
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            ious.append(float('nan'))  # Ignore this class
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


# Training Function
def train_model(model, train_dataloader, val_dataloader, device, num_classes=21, epochs=20, lr=1e-3,
                save_dir="./models"):
    """
    Train a segmentation model with MSE loss.
    """
    # Define optimizer and MSE loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Move model to device
    model.to(device)

    # Best model tracking
    best_miou = 0.0
    best_model_path = os.path.join(save_dir, "best_model.pth")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0

        # Training phase
        for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images, targets = images.to(device), targets.to(device)

            # Forward pass
            outputs = model(images)
            # Convert targets to one-hot encoding for MSE
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            loss = criterion(outputs, targets_one_hot)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            num_batches += 1

        train_loss /= len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_miou = None  # Only calculated every 5 epochs
        num_batches = 0

        with torch.no_grad():
            for images, targets in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images, targets = images.to(device), targets.to(device)

                # Forward pass
                outputs = model(images)
                # Convert targets to one-hot encoding for MSE
                targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
                loss = criterion(outputs, targets_one_hot)

                # Accumulate metrics
                val_loss += loss.item()
                num_batches += 1

            val_loss /= len(val_dataloader)

            # Calculate mIoU every 5 epochs
            if (epoch + 1) % 5 == 0:
                val_miou = 0.0
                num_batches = 0
                for images, targets in tqdm(val_dataloader, desc="Calculating mIoU"):
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    batch_miou = calculate_miou(preds, targets, num_classes)
                    val_miou += batch_miou if not np.isnan(batch_miou) else 0
                    num_batches += 1
                val_miou /= num_batches

        # Update learning rate
        scheduler.step(val_loss)

        # Save the best model
        if val_miou is not None and val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with mIoU: {best_miou:.4f}")

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        if val_miou is not None:
            print(f"  Val mIoU:   {val_miou:.4f}")

    # Save the final model
    final_model_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    print("Training completed.")
    return model


# Training augmentations
def get_train_augmentations():
    return Compose([
        Resize(height=256, width=256),
        HorizontalFlip(p=0.5),
        RandomCrop(height=256, width=256),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        GaussianBlur(blur_limit=(3, 5), p=0.3),
        GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# Validation preprocessing
def get_val_preprocessing():
    return Compose([
        Resize(height=256, width=256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# Training preprocessing function
def train_preprocess(image, target):
    augmentations = get_train_augmentations()
    augmented = augmentations(image=np.array(image), mask=np.array(target))
    image = augmented['image']
    target = torch.as_tensor(augmented['mask'], dtype=torch.long)
    return image, target


# Validation preprocessing function
def val_preprocess(image, target):
    preprocess = get_val_preprocessing()
    processed = preprocess(image=np.array(image), mask=np.array(target))
    image = processed['image']
    target = torch.as_tensor(processed['mask'], dtype=torch.long)
    return image, target


# Main training script
if __name__ == "__main__":
    from FinalModel import AdvancedSegmentationCNN
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    num_classes = 21
    model = AdvancedSegmentationCNN(num_classes)

    # Load PASCAL VOC 2012 dataset
    train_dataset = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="train",
        download=True,
        transforms=train_preprocess
    )

    val_dataset = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="val",
        download=True,
        transforms=val_preprocess
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        num_classes=num_classes,
        epochs=30,
        lr=1e-3
    )
