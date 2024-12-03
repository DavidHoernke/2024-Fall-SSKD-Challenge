import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from tqdm import tqdm
import matplotlib.pyplot as plt

from FinalModel import AdvancedSegmentationCNN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
def preprocess(image, target):
    transforms = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    image = transforms(image)
    target = torch.as_tensor(np.array(target.resize((256, 256), resample=0)), dtype=torch.long)
    return image, target

# Knowledge distillation training
def train_knowledge_distillation(teacher, student, train_loader, val_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, save_path):
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    student.train()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_logits = teacher(inputs)['out']

            # Student forward pass
            student_logits = student(inputs)

            # Soften logits for KD
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=1)

            # Calculate KD loss
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size(0) * (T**2)

            # Calculate CE loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted loss
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        val_loss = 0.0
        student.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                student_logits = student(inputs)
                label_loss = ce_loss(student_logits, labels)
                val_loss += label_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader)}, Val Loss: {avg_val_loss}")

        # Save the best model weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), save_path)
            print(f"Validation loss improved to {avg_val_loss:.4f}. Model weights saved!")


        torch.save(student.state_dict(), "LATEST"+save_path)
        # Visualize one random prediction
        visualize_prediction(student, val_loader, device)

# Visualization function
def visualize_prediction(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            target_np = targets[0].cpu().numpy()
            prediction_np = predictions[0]

            # Visualize input, ground truth, and prediction
            input_img = images[0].permute(1, 2, 0).cpu().numpy()
            input_img = (input_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            input_img = np.clip(input_img, 0, 1)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(input_img)
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(target_np, cmap="tab20")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(prediction_np, cmap="tab20")
            plt.title("Prediction")
            plt.axis("off")

            plt.show()
            break  # Visualize only one random image per epoch

# Main script
if __name__ == "__main__":
    # Load datasets
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
    student = AdvancedSegmentationCNN(num_classes=21).to(device)
    student.load_state_dict(torch.load("HeavyAugment_segmentation_model_best_augmented1.pth"))

    # Train student model with KD
    train_knowledge_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=1e-4,
        T=2,
        soft_target_loss_weight=0.3,
        ce_loss_weight=0.7,
        device=device,
        save_path="Heavy_T7_Soft_0.7_best_student_model.pth"
    )
