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

def walk_the_dir(dirname):
    for dirpath, dirnames, filenames in os.walk(dirname):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# walk_the_dir(FOOD_PATH / "test")
# walk_the_dir(FOOD_PATH / "train")

test_path = FOOD_PATH / "test"
train_path = FOOD_PATH / "train"

# create dataset
from torchvision.datasets import ImageFolder

transformer = torchvision.transforms.Compose([
    # Resize the image to 64x64 pixels
    torchvision.transforms.Resize((64, 64)),
    # Random horizontal flipping
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    # Convert the image to a tensor with pixels in the range [0, 1]
    torchvision.transforms.ToTensor(),
])

train_data = ImageFolder(train_path, transform=transformer)
test_data = ImageFolder(test_path, transform=transformer)

print(train_data)
print(test_data)

print("Test classes:", test_data.class_to_idx)
print("Train classes:", train_data.class_to_idx)

# create dataloader
from torch.utils.data import DataLoader

train_data_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=10, shuffle=True)

print("Train data loader:", train_data_loader)
print("Test data loader:", test_data_loader)

# seed 42
torch.manual_seed(42)

# # pick a random image from the training data
# random_image = random.randint(0, len(train_data) - 1)
# sample_image, sample_label = train_data[random_image]

# print(train_data[random_image])

# # convert image to numpy array
# sample_image_array = np.array(sample_image)

# print(sample_image_array.shape)

# # plot the image
# plt.imshow(sample_image_array.T)
# plt.title(f"Index: {random_image}, Class: {sample_label}")
# plt.axis(False);
# plt.show()

# Model 1: Build a CNN model