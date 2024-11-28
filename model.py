import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # Ensure you have `torchsummary` installed: pip install torchsummary

# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# Improved SimpleSegmentationCNN Model
class SimpleSegmentationCNNImproved(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationCNNImproved, self).__init__()

        # Encoder (Downsampling with Depthwise Separable Convolutions)
        self.enc_conv1 = DepthwiseSeparableConv(3, 64)
        self.enc_conv2 = DepthwiseSeparableConv(64, 128)
        self.pool = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = DepthwiseSeparableConv(128, 256)

        # Decoder (Upsampling with Skip Connections)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            DepthwiseSeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv1(x))
        x2 = self.pool(F.relu(self.enc_conv2(x1)))

        # Bottleneck
        x3 = F.relu(self.bottleneck(x2))

        # Decoder with skip connections
        x4 = F.relu(self.upconv1(x3))
        x4 = x4 + F.interpolate(x2, size=x4.shape[2:], mode="bilinear", align_corners=False)  # Align dimensions
        x5 = F.relu(self.dec_conv1(x4))
        x6 = F.relu(self.upconv2(x5))
        x6 = x6 + F.interpolate(x1, size=x6.shape[2:], mode="bilinear", align_corners=False)  # Align dimensions
        x7 = F.relu(self.dec_conv2(x6))

        # Final output
        output = self.final_conv(x7)
        output = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
        return output


# Test the model with torchsummary
if __name__ == "__main__":
    num_classes = 21  # Example for VOC dataset
    model = SimpleSegmentationCNNImproved(num_classes)
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available

    # Print the model summary
    print(summary(model, input_size=(3, 256, 256)))  # Input size: (channels, height, width)
