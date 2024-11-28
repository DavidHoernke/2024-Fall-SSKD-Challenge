import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.datasets import VOCSegmentation
from PIL import Image
from torchvision.transforms import ToTensor


def preprocess(image, mask):
    from torchvision.transforms import ToTensor
    from PIL import Image
    import numpy as np
    import torch

    # Resize the image (bilinear interpolation)
    image = image.resize((256, 256), resample=Image.BILINEAR)
    image = ToTensor()(image)  # Convert the image to a tensor, scales to [0, 1]

    # Resize the mask (strict nearest-neighbor interpolation for labels)
    mask = mask.resize((256, 256), resample=Image.BILINEAR)

    # # Convert the mask to a numpy array, ensuring it retains integer values
    # mask = np.array(mask, dtype=np.int64)

    # Convert the mask to a PyTorch tensor without any scaling
    mask = torch.as_tensor(mask, dtype=torch.long)

    return image, mask



# Load dataset without any preprocessing
raw_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="train",
    download=True
)

vraw_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="val",
    download=True
)

tvraw_dataset = VOCSegmentation(
    root="./data",
    year="2012",
    image_set="trainval",
    download=True
)
print(f"Number of samples in the dataset: {len(raw_dataset)}")
print(f"Number of samples in the val dataset: {len(vraw_dataset)}")
print(f"Number of samples in the train-val dataset: {len(tvraw_dataset)}")




