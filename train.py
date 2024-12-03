import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, HorizontalFlip, RandomCrop, Normalize, Resize, ColorJitter, GaussianBlur, \
    GridDistortion, ElasticTransform, CoarseDropout
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import ToTensor
from tqdm import tqdm

from Adv3Model import CompactResNetSegmentation
from KnowledgeDistillation.BetterTrain import train_knowledge_distillation, preprocess

# # Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your custom model
from FinalModel import AdvancedSegmentationCNN  # Adjust this based on your file structure
num_classes = 21  # VOCSegmentation has 21 classes including background
model = AdvancedSegmentationCNN(num_classes).to(device)
#
# # Define augmentations and preprocessing
# def get_augmentations():
#     return Compose([
#         Resize(height=256, width=256),  # Resize to ensure minimum dimensions
#         HorizontalFlip(p=0.5),  # Random horizontal flip
#         RandomCrop(height=256, width=256),  # Crop to 256x256
#         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2()  # Convert to PyTorch tensors
#     ])
#
# def preprocess(image, target):
#     augmentations = get_augmentations()
#     augmented = augmentations(image=np.array(image), mask=np.array(target))
#     image = augmented['image']
#     target = torch.as_tensor(augmented['mask'], dtype=torch.long)
#     return image, target
#
# # Load PASCAL VOC 2012 dataset
# train_dataset = VOCSegmentation(
#     root="./data",
#     year="2012",
#     image_set="train",
#     download=True,
#     transforms=preprocess
# )
#
# val_dataset = VOCSegmentation(
#     root="./data",
#     year="2012",
#     image_set="val",
#     download=True,
#     transforms=preprocess  # No augmentations for validation
# )
#
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
#
# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the "void" class
# optimizer = Adam(model.parameters(), lr=0.001)
#
# # Training settings
# epochs = 200
# patience = 30  # Early stopping patience
# best_val_loss = float('inf')
# early_stop_counter = 0
# train_losses, val_losses = [], []
# best_model_path = "best_advanced3_segmentation_cnn.pth"
# scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
# # Training loop
# for epoch in range(epochs):
#     start_time = time.time()
#     model.train()
#     epoch_train_loss = 0
#     for images, targets in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
#         images, targets = images.to(device), targets.to(device)
#
#         # Forward pass
#         outputs = model(images)
#
#         # Calculate loss
#         loss = criterion(outputs, targets)
#
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         epoch_train_loss += loss.item()
#
#     avg_train_loss = epoch_train_loss / len(train_dataloader)
#     train_losses.append(avg_train_loss)
#
#     # Validation phase
#     model.eval()
#     epoch_val_loss = 0
#     with torch.no_grad():
#         for images, targets in tqdm(val_dataloader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):
#             images, targets = images.to(device), targets.to(device)
#
#             outputs = model(images)
#             loss = criterion(outputs, targets)
#             epoch_val_loss += loss.item()
#
#     avg_val_loss = epoch_val_loss / len(val_dataloader)
#     val_losses.append(avg_val_loss)
#
#     scheduler.step()
#
#     # Save the best model if validation loss improves
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(model.state_dict(), best_model_path)
#         print(f"Epoch {epoch + 1}: Validation loss improved to {avg_val_loss:.4f}. Model saved.")
#         early_stop_counter = 0  # Reset counter if loss improves
#     else:
#         early_stop_counter += 1  # Increment counter if no improvement
#         print(f"Epoch {epoch + 1}: Validation loss did not improve. Patience counter: {early_stop_counter}/{patience}")
#
#     # Early stopping
#     if early_stop_counter >= patience:
#         print(f"Early stopping triggered. No improvement in validation loss for {patience} epochs.")
#         break
#
#     # Print epoch summary
#     epoch_time = time.time() - start_time
#     print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
#           f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
#
# # Plot the loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.legend()
# plt.savefig("loss_plot3.png")
# plt.show()
#
# print("Training complete. Best model saved as:", best_model_path)
#
#


#BARRIER

# Model setup
total_classes = 21  # VOCSegmentation has 21 classes including background
segmentation_model = CompactResNetSegmentation(total_classes).to(device)

# Augmentations
def define_augments():
    return Compose([
        Resize(height=256, width=256),  # Resize images
        RandomCrop(height=256, width=256),  # Random crop
        HorizontalFlip(p=0.5),  # Horizontal flip
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.7),  # Color jitter
        GaussianBlur(blur_limit=(3, 8), p=0.4),  # Gaussian blur
        GridDistortion(num_steps=6, distort_limit=0.4, p=0.4),  # Grid distortions
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),  # Elastic transformations
        CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.4),  # Coarse dropout
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalization
        ToTensorV2()  # Convert to PyTorch tensors
    ]) #These got me 0.195 gonna try more aggressive ones next

# Preprocessing
def preprocess_images_and_targets(img, mask):
    augmentation_pipeline = define_augments()
    augmented_data = augmentation_pipeline(image=np.array(img), mask=np.array(mask))
    processed_image = augmented_data['image']
    processed_mask = torch.as_tensor(augmented_data['mask'], dtype=torch.long)
    return processed_image, processed_mask

# Dataset and Dataloader setup
train_set = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transforms=preprocess_images_and_targets
)

validation_set = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="val",
    download=True,
    transforms=preprocess_images_and_targets  # No augmentations for validation
)

training_loader = DataLoader(train_set, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=16, shuffle=False)

# Loss, Optimizer, and Scheduler
loss_function = nn.CrossEntropyLoss(ignore_index=255)  # Ignore void class
optimizer_algo = Adam(segmentation_model.parameters(), lr=0.001)
learning_scheduler = StepLR(optimizer_algo, step_size=30, gamma=0.5)

# Training parameters
max_epochs = 200
early_stop_threshold = 30  # Early stopping threshold
lowest_validation_loss = float('inf')
early_stop_counter = 0
training_losses, validation_losses = [], []
model_save_path = "ADv3_ResNetStyle_UltraLight.pth"

# Training Loop
for current_epoch in range(max_epochs):
    epoch_start_time = time.time()
    segmentation_model.train()
    epoch_training_loss = 0

    for batch_images, batch_targets in tqdm(training_loader, desc=f"Epoch {current_epoch + 1}/{max_epochs} - Training"):
        batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)

        # Forward pass and loss calculation
        predictions = segmentation_model(batch_images)
        batch_loss = loss_function(predictions, batch_targets)

        # Backpropagation
        optimizer_algo.zero_grad()
        batch_loss.backward()
        optimizer_algo.step()

        epoch_training_loss += batch_loss.item()

    avg_training_loss = epoch_training_loss / len(training_loader)
    training_losses.append(avg_training_loss)

    # Step scheduler
    learning_scheduler.step()

    # Validation phase
    segmentation_model.eval()
    epoch_validation_loss = 0

    with torch.no_grad():
        for val_images, val_targets in tqdm(validation_loader, desc=f"Epoch {current_epoch + 1}/{max_epochs} - Validation"):
            val_images, val_targets = val_images.to(device), val_targets.to(device)

            val_predictions = segmentation_model(val_images)
            val_loss = loss_function(val_predictions, val_targets)
            epoch_validation_loss += val_loss.item()

    avg_validation_loss = epoch_validation_loss / len(validation_loader)
    validation_losses.append(avg_validation_loss)

    # Save best model
    if avg_validation_loss < lowest_validation_loss:
        lowest_validation_loss = avg_validation_loss
        torch.save(segmentation_model.state_dict(), model_save_path)
        print(f"Epoch {current_epoch + 1}: Validation loss improved to {avg_validation_loss:.4f}. Model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"Epoch {current_epoch + 1}: Validation loss did not improve. Patience counter: {early_stop_counter}/{early_stop_threshold}")

    # Early stopping
    if early_stop_counter >= early_stop_threshold:
           print(f"Early stopping activated. Validation loss not improved for {early_stop_threshold} epochs.")
        break

    # Epoch summary
    epoch_end_time = time.time()
    print(f"Epoch {current_epoch + 1}/{max_epochs}: Train Loss: {avg_training_loss:.4f}, "
          f"Validation Loss: {avg_validation_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_curve_augmented_model.png")
plt.show()

print("Training complete. Best model saved at:", model_save_path)

student = CompactResNetSegmentation(num_classes=21)
student.load_state_dict(torch.load("ADv3_ResNetStyle_UltraLight.pth"))

def preprocess(image, target):
    transforms = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = transforms(image)
    target = torch.as_tensor(np.array(target.resize((256, 256), resample=0)), dtype=torch.long)
    return image, target

train_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transforms=preprocess
)
val_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="val",
    download=True,
    transforms=preprocess
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

# Initialize models
teacher = fcn_resnet50(weights="COCO_WITH_VOC_LABELS_V1").to(device)
# student = CompactResNetSegmentation(num_classes=21).to(device)

train_knowledge_distillation(
    teacher=teacher,
    student=student,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    learning_rate=1e-4,
    T=5,
    soft_target_loss_weight=0.5,
    ce_loss_weight=0.5,
    device=device,
    save_path="ResNetStyle_T5SOFT0.5best_student_model.pth"
)