import torch
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Adv3Model import CompactResNetSegmentation
from model import SimpleSegmentationCNNImproved  # Adjust this based on your file structure


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define fixed size for resizing
fixed_size = (256, 256)

# Define preprocessing transforms
def preprocess(image, target):
    """
    Preprocess the image and target mask.
    - Resize to a fixed size.
    - Normalize image and convert to tensors.
    """
    transforms = Compose([
        Resize(fixed_size),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = transforms(image)
    target = torch.as_tensor(np.array(target.resize(fixed_size, resample=0)), dtype=torch.long)
    return image, target

# Load PASCAL VOC 2012 dataset (validation set)
val_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="val",
    download=True,
    transforms=preprocess
)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load your trained model
num_classes = 21  # VOCSegmentation has 21 classes including background
model = CompactResNetSegmentation(num_classes)
# model.load_state_dict(torch.load("HeavyAugment_segmentation_model_best_augmented.pth", map_location=device)) 0.1912 miou
# model.load_state_dict(torch.load("segmentation_model_best_augmented.pth")) Mean Intersection over Union (mIoU): 0.1954
model.load_state_dict(torch.load("./KnowledgeDistillation/best_student_model.pth"))
model = model.to(device)



# Custom mIoU calculation function
def calculate_miou(pred, target, num_classes=21):
    """
    Calculate mean Intersection over Union (mIoU) for semantic segmentation.
    Args:
        pred (torch.Tensor): The predicted segmentation output (logits or probabilities), shape (H, W).
        target (torch.Tensor): The ground truth segmentation mask, shape (H, W).
        num_classes (int): The number of segmentation classes.
    Returns:
        float: Mean Intersection over Union (mIoU).
    """
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()
    ious = []

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()

        if union == 0:
            ious.append(float('nan'))  # Ignore classes with no ground truth
        else:
            ious.append(intersection / union)

    return np.nanmean(ious)


# Test method to calculate mIoU
def test_model(model, dataloader, device, num_classes=21):
    """
    Test the model and calculate mean IoU using custom calculate_miou function.
    Args:
    - model: Trained PyTorch model.
    - dataloader: DataLoader for the test/validation dataset.
    - device: Device to perform computation on ('cuda' or 'cpu').
    - num_classes: Number of classes for segmentation.
    Returns:
    - miou: Mean IoU across all samples and classes.
    """
    model.eval()  # Set model to evaluation mode
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)  # Move to device
            outputs = model(images)  # Forward pass
            batch_iou = calculate_miou(outputs, targets, num_classes=num_classes)  # Custom mIoU calculation

            if not np.isnan(batch_iou):  # Skip NaN results
                total_iou += batch_iou
                num_batches += 1

    # Average IoU across all batches
    miou = total_iou / num_batches if num_batches > 0 else 0.0
    print(f"Mean Intersection over Union (mIoU): {miou:.4f}")
    return miou


# Visualization function for predictions
def visualize_predictions(model, dataloader, device, num_samples=20):
    """
    Visualize a few predictions from the model.
    Args:
    - model: Trained PyTorch model.
    - dataloader: DataLoader for the test/validation dataset.
    - device: Device to perform computation on ('cuda' or 'cpu').
    - num_samples: Number of samples to visualize.
    """
    model.eval()
    examples_shown = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Visualize only a few samples
            if examples_shown < num_samples:
                # Denormalize image for visualization
                image_np = images[0].cpu().permute(1, 2, 0).numpy()
                image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                image_np = np.clip(image_np, 0, 1)

                target_np = targets[0].cpu().numpy()
                pred_np = preds[0].cpu().numpy()

                # Plot the images
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image_np)
                plt.title("Input Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(target_np, cmap="tab20")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pred_np, cmap="tab20")
                plt.title("Model Prediction")
                plt.axis("off")

                plt.show()

                examples_shown += 1


# Start evaluation
print("Starting evaluation...")
test_miou = test_model(model, val_dataloader, device, num_classes=num_classes)
print(f"Final mIoU on validation set: {test_miou:.4f}")

# Visualize predictions
print("Visualizing predictions...")
visualize_predictions(model, val_dataloader, device, num_samples=3)
