import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
from torchmetrics import JaccardIndex
import numpy as np
import matplotlib.pyplot as plt
import random

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed input size for resizing
fixed_size = (256, 256)

# Define preprocessing transforms
def preprocess(image, target):
    transforms = Compose([
        Resize(fixed_size),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = transforms(image)
    target = torch.as_tensor(np.array(target.resize(fixed_size, resample=0)), dtype=torch.long)
    return image, target

# Load Pascal VOC 2012 dataset
val_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="val",
    download=True,
    transforms=preprocess
)

# # Create DataLoader for validation dataset
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#
# # Load pretrained FCN_ResNet50 model
model = fcn_resnet50(weights="COCO_WITH_VOC_LABELS_V1").to(device)
model.eval()  # Set model to evaluation mode

# mIoU calculation function
def calculate_miou(model, dataloader, device, num_classes=21):
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Calculating mIoU"):
            images, targets = images.to(device), targets.to(device)

            # Exclude void regions (255)
            mask = targets != 255
            valid_targets = targets[mask]
            if valid_targets.numel() == 0:
                continue

            outputs = model(images)
            if isinstance(outputs, dict) and "out" in outputs:
                outputs = outputs["out"]  # Use "out" key if present
            preds = torch.argmax(outputs, dim=1)
            valid_preds = preds[mask]

            # Calculate IoU for this batch
            try:
                batch_iou = jaccard(valid_preds, valid_targets)
                total_iou += batch_iou.item()
                num_batches += 1
            except RuntimeError as e:
                print(f"Error calculating IoU for batch: {e}")

    miou = total_iou / num_batches if num_batches > 0 else 0.0
    print(f"Mean Intersection over Union (mIoU): {miou:.4f}")
    return miou

# Visualization function for predictions
def visualize_predictions(model, dataset, device, num_samples=5):
    """
    Visualize a few random predictions from the model.
    Args:
    - model: Trained segmentation model.
    - dataset: Dataset object (e.g., VOCSegmentation).
    - device: Device to perform computation on ('cuda' or 'cpu').
    - num_samples: Number of samples to visualize.
    """
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        # Load a random sample
        image, target = dataset[idx]
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        target = target.to(device)

        with torch.no_grad():
            outputs = model(image)
            if isinstance(outputs, dict) and "out" in outputs:
                outputs = outputs["out"]  # Use "out" key if present
            pred = torch.argmax(outputs, dim=1).squeeze(0)  # Get predicted mask

        # Denormalize the image for visualization
        image_np = image[0].cpu().permute(1, 2, 0).numpy()
        image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image_np = np.clip(image_np, 0, 1)

        # Convert target and prediction to numpy arrays
        target_np = target.cpu().numpy()
        pred_np = pred.cpu().numpy()

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

# Run evaluation and visualization
if __name__ == "__main__":
    # from AdvModel import AdvancedSegmentationCNN  # Import your custom model
    #
    # # Load your custom model
    # num_classes = 21  # Number of classes in VOCSegmentation
    # model = AdvancedSegmentationCNN(num_classes).to(device)
    #
    # # Load the trained weights
    # model.load_state_dict(torch.load("best_advanced_segmentation_cnn.pth", map_location=device))
    # model.eval()  # Set model to evaluation mode

    print("Starting evaluation...")
    miou = calculate_miou(model, val_dataloader, device, num_classes=21)
    print(f"Final mIoU on Pascal VOC 2012 validation set: {miou:.4f}")

    print("Visualizing predictions...")
    visualize_predictions(model, val_dataset, device, num_samples=5)

