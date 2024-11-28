import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
from Adv3Model import CompactResNetSegmentation
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import numpy as np

# Define Knowledge Distillation Loss
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

# Sanity Check Function
def sanity_checks(data_loader, teacher, student, device):
    teacher.eval()
    student.eval()

    loss_fn = KnowledgeDistillationLoss(alpha=0.5, temperature=3)

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            teacher_logits = teacher(images)['out']
        student_logits = student(images)

        # Compute Loss
        total_loss, ce_loss, kl_loss = loss_fn(student_logits, teacher_logits, targets)

        # Debug Information
        print(f"Teacher logits shape: {teacher_logits.shape}")
        print(f"Student logits shape: {student_logits.shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Teacher logits range: {teacher_logits.min().item()}, {teacher_logits.max().item()}")
        print(f"Student logits range: {student_logits.min().item()}, {student_logits.max().item()}")
        print(f"Targets unique values: {targets.unique()}")
        print(f"CE Loss: {ce_loss.item()}")
        print(f"KL Loss: {kl_loss.item()}")
        print(f"Total Loss: {total_loss.item()}")
        break

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

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)

    # Run sanity checks
    sanity_checks(train_loader, teacher, student, device)
