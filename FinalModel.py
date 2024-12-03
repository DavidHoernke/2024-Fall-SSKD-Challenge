import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Depthwise Separable Convolution
class ResidualDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(ResidualDepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.residual(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x + residual)

# Updated AdvancedSegmentationCNN Model
class AdvancedSegmentationCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedSegmentationCNN, self).__init__()

        # Encoder (Downsampling with Residual Connections)
        self.enc_conv1 = nn.Sequential(
            ResidualDepthwiseSeparableConv(3, 64),
            ResidualDepthwiseSeparableConv(64, 64)
        )
        self.enc_conv2 = nn.Sequential(
            ResidualDepthwiseSeparableConv(64, 128),
            ResidualDepthwiseSeparableConv(128, 128)
        )
        self.enc_conv3 = nn.Sequential(
            ResidualDepthwiseSeparableConv(128, 256),
            ResidualDepthwiseSeparableConv(256, 256)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualDepthwiseSeparableConv(256, 512, dilation=2, padding=2),
            nn.Dropout(0.5),  # Dropout for regularization
            ResidualDepthwiseSeparableConv(512, 512, dilation=4, padding=4)
        )

        # Decoder (Upsampling with Residual Connections)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            ResidualDepthwiseSeparableConv(512, 256),
            ResidualDepthwiseSeparableConv(256, 256)
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            ResidualDepthwiseSeparableConv(256, 128),
            ResidualDepthwiseSeparableConv(128, 128)
        )
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            ResidualDepthwiseSeparableConv(128, 64),
            ResidualDepthwiseSeparableConv(64, 64)
        )

        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        # Store intermediate features for KD
        self.features = None

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)  # Output: (batch, 64, H, W)
        x2 = self.enc_conv2(self.pool(x1))  # Output: (batch, 128, H/2, W/2)
        x3 = self.enc_conv3(self.pool(x2))  # Output: (batch, 256, H/4, W/4)

        # Save intermediate feature maps for KD
        self.features = x3

        # Bottleneck
        x4 = self.bottleneck(self.pool(x3))  # Output: (batch, 512, H/8, W/8)

        # Decoder with skip connections
        x5 = self.upconv1(x4)  # Output: (batch, 256, H/4, W/4)
        x5 = torch.cat([x5, x3], dim=1)  # Concatenate skip connection
        x5 = self.dec_conv1(x5)

        x6 = self.upconv2(x5)  # Output: (batch, 128, H/2, W/2)
        x6 = torch.cat([x6, x2], dim=1)  # Concatenate skip connection
        x6 = self.dec_conv2(x6)

        x7 = self.upconv3(x6)  # Output: (batch, 64, H, W)
        x7 = torch.cat([x7, x1], dim=1)  # Concatenate skip connection
        x7 = self.dec_conv3(x7)

        # Final output
        output = self.final_conv(x7)
        output = F.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=False)
        return output


# Test the updated model
if __name__ == "__main__":
    num_classes = 21  # Example for Pascal VOC dataset
    model = AdvancedSegmentationCNN(num_classes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Print the model summary
    from torchsummary import summary
    print(summary(model, input_size=(3, 256, 256)))
