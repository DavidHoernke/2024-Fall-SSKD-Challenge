from torchsummary import summary
from unet_model import UNet  # Replace with your U-Net implementation

model = UNet(n_channels=1, n_classes=2)  # Example for binary segmentation
summary(model, input_size=(1, 572, 572))  # Replace with your input dimensions
