import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from tqdm import tqdm
import matplotlib.pyplot as plt
from Adv2Model import AdvancedSegmentationCNN

class FeatureExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self.hook = None
        self._register_hook()

    def _register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.hook = module.register_forward_hook(self._hook_fn)
                break

    def _hook_fn(self, module, input, output):
        self.features = output

    def remove_hook(self):
        if self.hook:
            self.hook.remove()

    def get_features(self):
        return self.features

# Cosine similarity loss
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()

    def forward(self, teacher_features, student_features):
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)
        student_flat = student_features.view(student_features.size(0), -1)
        return 1 - torch.nn.functional.cosine_similarity(teacher_flat, student_flat, dim=1).mean()

# Preprocessing
def preprocess(image, target):
    # Define transformations for the input image
    transforms = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Apply transforms to the image
    image = transforms(image)

    # Resize and convert the target (segmentation mask)
    target = target.resize((256, 256))  # Resize the target
    target = torch.as_tensor(np.array(target), dtype=torch.long)  # Convert to tensor

    return image, target


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

def train_knowledge_distillation_feature_based(
    teacher, student, train_loader, val_loader, epochs, learning_rate, feature_loss_weight, ce_loss_weight, device, save_path
):
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    cosine_loss = CosineSimilarityLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    # Attach a feature extractor to the teacher model
    teacher_extractor = FeatureExtractor(teacher, layer_name="backbone.layer4")  # Example layer

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
                teacher_outputs = teacher(inputs)['out']
                teacher_features = teacher_extractor.get_features()

            # Student forward pass
            student_outputs = student(inputs)
            student_features = student.features

            # Feature-based loss (Cosine similarity)
            feature_loss = cosine_loss(teacher_features, student_features)

            # Cross-entropy loss
            label_loss = ce_loss(student_outputs, labels)

            # Weighted loss
            loss = feature_loss_weight * feature_loss + ce_loss_weight * label_loss

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader)}")

        # Validation phase
        val_loss = 0.0
        student.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                student_outputs = student(inputs)
                label_loss = ce_loss(student_outputs, labels)
                val_loss += label_loss.item()

                # Visualize one prediction on the first batch
                if i == 0:
                    visualize_prediction(student, [(inputs, labels)], device)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(student.state_dict(), save_path)
            print(f"Validation loss improved to {avg_val_loss:.4f}. Model weights saved!")

    # Remove the hook after training
    teacher_extractor.remove_hook()


# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    teacher = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True).to(device)
    student = AdvancedSegmentationCNN(num_classes=21).to(device)

    train_knowledge_distillation_feature_based(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=1e-4,
        feature_loss_weight=0.5,
        ce_loss_weight=0.5,
        device=device,
        save_path="student_model.pth"
    )
