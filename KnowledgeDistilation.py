import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from tqdm import tqdm

from Adv2Model import AdvancedSegmentationCNN

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 8
learning_rate = 1e-4
num_epochs = 50
alpha = 0.7
temperature = 4.0
num_classes = 21  # Pascal VOC 2012 has 21 classes (including background)
best_model_path = "best_student_model.pth"


def preprocess(image, target):
    """
    Preprocess the image and segmentation mask.
    - Resize the image and mask to (256, 256).
    - Normalize the image and convert both to tensors.
    """
    transforms = Compose([
        Resize((256, 256)),  # Resize both image and target
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Normalize the image
    ])

    # Ensure the target is a PIL image and resize using nearest neighbor for masks
    if not isinstance(target, Image.Image):
        raise TypeError("Expected target to be a PIL Image")
    target = target.resize((256, 256), resample=Image.NEAREST)

    # Apply transformations
    image = transforms(image)
    target = torch.as_tensor(np.array(target), dtype=torch.long)  # Convert mask to tensor
    return image, target


# Dataset and DataLoader
train_dataset = VOCSegmentation(
    root="./data", year="2012", image_set="train", download=True, transforms=preprocess
)
val_dataset = VOCSegmentation(
    root="./data", year="2012", image_set="val", download=True, transforms=preprocess
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Models
# Teacher: Pretrained FCN_ResNet50
teacher_model = fcn_resnet50(weights="COCO_WITH_VOC_LABELS_V1").to(device)
teacher_model.eval()  # Teacher remains frozen during training

# Student: Custom Model
from model import SimpleSegmentationCNNImproved  # Import your custom model

student_model = AdvancedSegmentationCNN(num_classes).to(device)

# Optimizer
optimizer = Adam(student_model.parameters(), lr=learning_rate)


# Loss function
def kd_loss(student_logits, teacher_logits, targets, alpha=0.7, temperature=4.0):
    """
    Knowledge Distillation Loss combining hard labels and soft labels.
    """
    # Hard label loss (CrossEntropy)
    hard_loss = F.cross_entropy(student_logits, targets, ignore_index=255)

    # Soft label loss (KL Divergence)
    teacher_soft = F.log_softmax(teacher_logits / temperature, dim=1)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

    return alpha * soft_loss + (1 - alpha) * hard_loss


# Validation: Compute mIoU
def validate_model(model, dataloader, device, num_classes=21):
    """
    Validate the model and compute mIoU using torchmetrics.JaccardIndex.
    """
    model.eval()
    jaccard = JaccardIndex(task="multiclass", num_classes=num_classes).to(device)
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images, targets = images.to(device), targets.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            mask = targets != 255
            valid_preds = preds[mask]
            valid_targets = targets[mask]
            if valid_targets.numel() == 0:
                continue

            batch_iou = jaccard(valid_preds, valid_targets)
            total_iou += batch_iou.item()
            num_batches += 1

    return total_iou / num_batches if num_batches > 0 else 0.0


# Training loop
best_val_miou = 0.0
for epoch in range(num_epochs):
    start_time = time.time()
    student_model.train()
    epoch_loss = 0.0

    for images, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        student_logits = student_model(images)
        with torch.no_grad():
            teacher_logits = teacher_model(images)["out"]

        # Compute knowledge distillation loss
        loss = kd_loss(student_logits, teacher_logits, targets, alpha=alpha, temperature=temperature)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Validate the model
    val_miou = validate_model(student_model, val_dataloader, device, num_classes)
    epoch_time = time.time() - start_time

    # Save the best model
    if val_miou > best_val_miou:
        best_val_miou = val_miou
        torch.save(student_model.state_dict(), best_model_path)
        print(f"Epoch {epoch + 1}: Validation mIoU improved to {val_miou:.4f}. Model saved.")

    # Log progress
    print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(train_dataloader):.4f}, "
          f"Validation mIoU: {val_miou:.4f}, Time: {epoch_time:.2f}s")

print(f"Training complete. Best mIoU: {best_val_miou:.4f}. Best model saved at {best_model_path}.")
