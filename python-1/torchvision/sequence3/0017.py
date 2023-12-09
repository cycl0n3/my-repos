import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("PyTorch Version: ", torch.__version__)

import torchvision
import torchvision.transforms as transforms
import torchmetrics

print("Torchvision Version: ", torchvision.__version__)
print("Torchmetrics Version: ", torchmetrics.__version__)

import requests
import zipfile

import os
from pathlib import Path

print("Requests Version: ", requests.__version__)

# Download the dataset
DATA_PATH = Path("data")
FOOD_ZIP_PATH = DATA_PATH / "pizza_steak_sushi.zip"
FOOD_PATH = DATA_PATH / "pizza_steak_sushi"

if not FOOD_PATH.exists():
    print("Downloading the data...")
    r = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    with open(FOOD_ZIP_PATH, "wb") as f:
        f.write(r.content)
    print("Unzipping the data...")
    with zipfile.ZipFile(FOOD_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(FOOD_PATH)
    print("Done!")
else:
    print("The data has already been downloaded and extracted.")

# Check the data
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("darkgrid")

# Load classes
class_names = sorted(os.listdir(FOOD_PATH / "train"))
print("Class names:", class_names)

# convert class names to indexed map
class_names_to_index = dict((i, c) for i, c in enumerate(class_names))
print("Class names to index:", class_names_to_index)

test_path = FOOD_PATH / "test"
train_path = FOOD_PATH / "train"

# create dataset
from torchvision.datasets import ImageFolder

train_transformer = transforms.Compose([
    # Resize the image to 64x64 pixels
    transforms.Resize((64, 64)),
    # Random horizontal flipping
    # TRANSFORMS.RandomHorizontalFlip(p=0.5),
    # Trivial augment wide
    transforms.TrivialAugmentWide(num_magnitude_bins=8),
    # Convert the image to a tensor with pixels in the range [0, 1]
    transforms.ToTensor(),
])

test_transformer = transforms.Compose([
    # Resize the image to 224x224 pixels
    transforms.Resize((224, 224)),
    # Convert the image to a tensor with pixels in the range [0, 1]
    transforms.ToTensor(),
])

train_data = ImageFolder(train_path, transform=train_transformer)
test_data = ImageFolder(test_path, transform=test_transformer)

class_names = train_data.classes
class_names_to_index = train_data.class_to_idx

print("Class names:", class_names)
print("Class names to index:", class_names_to_index)

# Original image shape
print("Image shape:", train_data[0][0].shape)

# create dataloader
from torch.utils.data import DataLoader

shuffle = True

train_data_loader = DataLoader(train_data, batch_size=10, shuffle=shuffle)
test_data_loader = DataLoader(test_data, batch_size=10, shuffle=shuffle)

print("Length of train data loader:", len(train_data_loader))
print("Length of test data loader:", len(test_data_loader))

# seed 42
torch.manual_seed(42)

# plot some random training images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    random_idx = random.randint(0, len(train_data))
    sample_image, sample_label = train_data[random_idx]
    plt.imshow(sample_image.permute(1, 2, 0))
    plt.title(f"Index: {random_idx} - Label: {class_names[sample_label]}")
    plt.axis("off")
plt.show()

