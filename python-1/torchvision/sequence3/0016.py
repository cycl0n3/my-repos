import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("PyTorch Version: ", torch.__version__)

import torchvision
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

transformer1 = torchvision.transforms.Compose([
    # Resize the image to 224x224 pixels
    torchvision.transforms.Resize((224, 224)),
    # Convert the image to a tensor with pixels in the range [0, 1]
    torchvision.transforms.ToTensor(),
])

transformer2 = torchvision.transforms.Compose([
    # Resize the image to 64x64 pixels
    torchvision.transforms.Resize((64, 64)),
    # Random horizontal flipping
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # Convert the image to a tensor with pixels in the range [0, 1]
    torchvision.transforms.ToTensor(),
])

train_original_data = ImageFolder(train_path, transform=transformer1)
test_original_data = ImageFolder(test_path, transform=transformer1)

train_transformed_data = ImageFolder(train_path, transform=transformer2)
test_transformed_data = ImageFolder(test_path, transform=transformer2)

print(train_transformed_data)
print(test_transformed_data)

class_names = train_original_data.classes
class_names_to_index = train_original_data.class_to_idx

print("Class names:", class_names)
print("Class names to index:", class_names_to_index)

# Original image shape
print("Original image shape:", train_original_data[0][0].shape)

# Transformed image shape
print("Transformed image shape:", train_transformed_data[0][0].shape)

# create dataloader
from torch.utils.data import DataLoader

shuffle = True

train_original_data_loader = DataLoader(train_original_data, batch_size=10, shuffle=shuffle)
test_original_data_loader = DataLoader(test_original_data, batch_size=10, shuffle=shuffle)

# print("Train original data loader:", train_original_data_loader)
# print("Test original data loader:", test_original_data_loader)

train_transformed_data_loader = DataLoader(train_transformed_data, batch_size=10, shuffle=shuffle)
test_transformed_data_loader = DataLoader(test_transformed_data, batch_size=10, shuffle=shuffle)

# print("Train transformed data loader:", train_transformed_data_loader)
# print("Test transformed data loader:", test_transformed_data_loader)

# First train sample
print("First train sample:", train_original_data.samples[0])
print("First test sample:", test_original_data.samples[0])

# seed 42
torch.manual_seed(42)

# plot first image
image, label= train_transformed_data[0]
plt.imshow(image.permute(1, 2, 0))
plt.title(f"Label: {class_names[label]}")
plt.grid(False)
plt.axis(False)
plt.show()
