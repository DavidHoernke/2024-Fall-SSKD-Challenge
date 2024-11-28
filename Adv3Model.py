import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Residual Block
class CompactResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CompactResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.relu(out)

# Compact ResNet for 256x256 Input
class CompactResNetSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(CompactResNetSegmentation, self).__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample to 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual layers with fewer blocks
        self.layer1 = self._make_layer(16, 32, num_blocks=1, stride=2)  # 128x128 -> 64x64
        self.layer2 = self._make_layer(32, 64, num_blocks=1, stride=2)  # 64x64 -> 32x32
        self.layer3 = self._make_layer(64, 128, num_blocks=1, stride=2)  # 32x32 -> 16x16
        self.layer4 = self._make_layer(128, 256, num_blocks=1, stride=2)  # 16x16 -> 8x8

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 32x32 -> 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 64x64 -> 128x128
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)  # 128x128 -> 256x256
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(CompactResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(CompactResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)  # Initial convolution
        x = self.layer1(x)  # Residual block 1
        x = self.layer2(x)  # Residual block 2
        x = self.layer3(x)  # Residual block 3
        x = self.layer4(x)  # Residual block 4
        x = self.decoder(x)  # Decoder to full resolution
        return x

# Example Usage
if __name__ == "__main__":
    model = CompactResNetSegmentation(num_classes=21)  # 21 classes for Pascal VOC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Test model output
    input_tensor = torch.randn(1, 3, 256, 256).to(device)  # Batch size 1, 3 channels, 256x256
    output = model(input_tensor)
    print("Model output shape:", output.shape)  # Should be [1, 21, 256, 256]
    print("Total parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
