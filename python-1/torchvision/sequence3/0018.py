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

test_path = FOOD_PATH / "test"
train_path = FOOD_PATH / "train"

# create dataset
from torchvision.datasets import ImageFolder

simple_transform = transforms.Compose([
    # Resize the image to 64x64 pixels
    transforms.Resize((64, 64)),
    # Convert the image to a tensor with pixels in the range [0, 1]
    transforms.ToTensor(),
])

simple_train_data = ImageFolder(train_path, transform=simple_transform)
simple_test_data = ImageFolder(test_path, transform=simple_transform)

class_names = simple_train_data.classes
class_names_to_index = simple_train_data.class_to_idx

print("Class names:", class_names)
print("Class names to index:", class_names_to_index)

# create dataloader
from torch.utils.data import DataLoader

BATCH_SIZE = 10
NUM_WORKERS = int(os.cpu_count() / 2)

print("Number of workers:", NUM_WORKERS)

simple_train_data_loader = DataLoader(simple_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
simple_test_data_loader = DataLoader(simple_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Length of simple train data loader:", len(simple_train_data_loader))
print("Length of simple test data loader:", len(simple_test_data_loader))

# seed 42
torch.manual_seed(42)

# Model 0: TinyVGG without augmentation
class TinyVGG(nn.Module):
    def __init__(self, in_shape=3, hidden_units=10, out_shape=3):
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * hidden_units, out_shape),
        )
    def forward(self, x):
        x = self.conv_block_1(x); print("Conv block 1:", x.shape)
        x = self.conv_block_2(x); print("Conv block 2:", x.shape)
        x = self.classifier(x); print("Classifier:", x.shape)
        return x

from torchinfo import summary

demo_image = simple_train_data_loader.dataset[0]
print("Demo Image:", demo_image)

model = TinyVGG()
print("TinyVGG:", model)
print("TinyVGG summary:", summary(model, input_size=(1, 3, 64, 64), verbose=0))

demo_image_output = model(demo_image[0].unsqueeze(0))
print("TinyVGG output:", demo_image_output)

demo_image_softmax = F.softmax(demo_image_output, dim=1)
print("TinyVGG softmax output:", demo_image_softmax)

demo_image_pred = torch.argmax(demo_image_softmax, dim=1)
print("TinyVGG prediction:", demo_image_pred)

