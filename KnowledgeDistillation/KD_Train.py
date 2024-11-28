import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
from tqdm import tqdm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
from Adv3Model import CompactResNetSegmentation
import matplotlib.pyplot as plt
import numpy as np

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=5, kl_scale=0.001):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_scale = kl_scale
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=255)  # Ground truth loss
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")  # Teacher-Student divergence

    def forward(self, student_logits, teacher_logits, targets):
        # Normalize teacher logits
        teacher_logits = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)

        # Cross-Entropy Loss
        ce_loss = self.criterion_ce(student_logits, targets)

        # KL Divergence Loss
        teacher_soft = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        student_soft = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        kl_loss = self.criterion_kl(student_soft, teacher_soft) * (self.temperature ** 2)

        # Combine losses with scaling
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss * self.kl_scale
        return total_loss, ce_loss, kl_loss


# Data Augmentation
def define_transforms():
    return Compose([
        Resize(256, 256),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def preprocess_images_and_targets(img, mask):
    transform = define_transforms()
    augmented = transform(image=np.array(img), mask=np.array(mask))
    image = augmented['image']
    target = torch.as_tensor(augmented['mask'], dtype=torch.long)
    return image, target

# Training function
def train_with_distillation(student, teacher, train_loader, val_loader, device, epochs=50, lr=1e-4, alpha=0.5, save_path="best_student_model.pth"):
    teacher.eval()  # Freeze teacher model
    student.to(device)
    teacher.to(device)

    optimizer = optim.Adam(student.parameters(), lr=lr)
    distillation_loss = KnowledgeDistillationLoss(alpha=alpha, temperature=3)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training phase
        student.train()
        train_loss = 0.0
        ce_loss_total = 0.0
        kl_loss_total = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass for both teacher and student
            student_outputs = student(images)
            with torch.no_grad():
                teacher_outputs = teacher(images)['out']

            # Compute KD loss
            loss, ce_loss, kl_loss = distillation_loss(student_outputs, teacher_outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            ce_loss_total += ce_loss.item()
            kl_loss_total += kl_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        student.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                images, targets = images.to(device), targets.to(device)
                student_outputs = student(images)
                teacher_outputs = teacher(images)['out']
                loss, ce_loss, kl_loss = distillation_loss(student_outputs, teacher_outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), save_path)
            print(f"Validation loss improved to {avg_val_loss:.4f}. Model saved.")

        avg_ce_loss = ce_loss_total / len(train_loader)
        avg_kl_loss = kl_loss_total / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {avg_train_loss:.4f}, "
              f"CE Loss: {avg_ce_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("kd_loss_curve.png")
    plt.show()

# Main script
if __name__ == "__main__":
    # Initialize models
    teacher = fcn_resnet50(weights="COCO_WITH_VOC_LABELS_V1")
    student = CompactResNetSegmentation(num_classes=21)

    # Load datasets
    train_dataset = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="train",
        download=True,
        transforms=preprocess_images_and_targets
    )
    val_dataset = VOCSegmentation(
        root="./data",
        year="2012",
        image_set="val",
        download=True,
        transforms=preprocess_images_and_targets
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model with KD
    train_with_distillation(student, teacher, train_loader, val_loader, device, epochs=50, lr=1e-4, alpha=0.7)
