import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        """
        IoU Loss implementation for segmentation tasks.
        Args:
        - smooth: Smoothing constant to prevent division by zero.
        - ignore_index: The index to ignore in the loss computation.
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
        - logits: Model predictions before applying softmax (shape: [B, C, H, W]).
        - targets: Ground truth segmentation masks (shape: [B, H, W]).
        Returns:
        - IoU loss (scalar).
        """
        # Apply softmax to logits
        probs = torch.softmax(logits, dim=1)  # Shape: [B, C, H, W]

        # Mask out invalid target indices
        valid_mask = targets != self.ignore_index  # Shape: [B, H, W]
        targets = targets.clone()
        targets[~valid_mask] = 0  # Replace ignore index with valid class for one-hot encoding

        # One-hot encode the targets
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)  # Shape: [B, C, H, W]
        targets_one_hot = targets_one_hot * valid_mask.unsqueeze(1)  # Ignore invalid targets in one-hot encoding

        # Calculate intersection and union
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))  # Sum over H and W
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) - intersection

        # IoU coefficient
        iou = (intersection + self.smooth) / (union + self.smooth)

        # IoU loss
        return 1.0 - iou.mean()
