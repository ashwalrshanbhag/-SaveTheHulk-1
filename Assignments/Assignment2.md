
# Assignment: Semantic and Instance Segmentation

##  Objective
Implement two segmentation models:
1. **Semantic Segmentation** using U-Net or DeepLabv3+.
2. **Instance Segmentation** using Mask R-CNN.

Use appropriate Kaggle datasets and evaluate models with IoU, Pixel Accuracy, and Dice Score.

---

## Part 1: Semantic Segmentation - Oxford Pets

###  Dataset
**[Oxford Pets - Kaggle](https://www.kaggle.com/datasets/andrewmvd/oxford-pets-segmentation)**

###  Task
Segment pets from the background in each image.

###  Starter Code

```python
# Install dependencies
!pip install segmentation-models-pytorch

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
# This smp class has many prebuilt architectures that you can use.
# Try to implement a unet by urself for extra credit!

# Dataset Class
class OxfordPetsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.images[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, self.masks[idx])).convert("L")

        img = img.resize((256, 256))
        mask = mask.resize((256, 256))

        img = np.array(img) / 255.0
        mask = np.array(mask) > 0

        if self.transform:
            img = self.transform(img)

        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.long)

        return img, mask
```

###  Tasks
- Train a U-Net or DeepLabv3+ for at least 10 epochs.
- Use `CrossEntropyLoss` or `DiceLoss`.
- Evaluate with IoU and Pixel Accuracy.
- Visualize predictions vs. ground truth.

---

##  Part 2: Instance Segmentation - Carvana

###  Dataset
**[Carvana Image Masking Challenge - Kaggle](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data)**

###  Task
Segment individual cars (simulate instance segmentation with one object per image).

###  Starter Code

```python
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Dummy input for testing
dummy_input = torch.rand(1, 3, 256, 256)
output = model(dummy_input)
```

###  Tasks
- Preprocess masks as binary masks for car vs. background.
- Use torchvision’s `Mask R-CNN` or `Detectron2`.
- Train and evaluate with IoU and mask overlays.
- Visualize instance masks + bounding boxes.

---





##  Submission Instructions
- Upload to github with:
  - Scripts / notebooks
  - Visualizations and plots
  - Your understanding of the architecture and working of each model.

---

##  Deadline
**Next Weekend**
