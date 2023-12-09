import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd

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
    r = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", stream=True)
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

complex_transform = transforms.Compose([
    # Resize the image to 128x128 pixels
    transforms.Resize((64, 64)),
    
    # Normalize the image with the mean and standard deviation of the ImageNet dataset
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    # Some augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    # Convert the image to a tensor with pixels in the range [0, 1]
    transforms.ToTensor(),
])

simple_train_data = ImageFolder(train_path, transform=complex_transform)
simple_test_data = ImageFolder(test_path, transform=complex_transform)

class_names = simple_train_data.classes
class_names_to_index = simple_train_data.class_to_idx

print("Class names:", class_names)
print("Class names to index:", class_names_to_index)

# create dataloader
from torch.utils.data import DataLoader

BATCH_SIZE = 10

simple_train_data_loader = DataLoader(simple_train_data, batch_size=BATCH_SIZE, shuffle=True)
simple_test_data_loader = DataLoader(simple_test_data, batch_size=BATCH_SIZE, shuffle=False)

print("Length of simple train data loader:", len(simple_train_data_loader))
print("Length of simple test data loader:", len(simple_test_data_loader))

# seed 42
torch.manual_seed(42)

# Model 0: TinyVGG without augmentation
class TinyVGG(nn.Module):
    def __init__(self, in_shape, hidden_units, out_shape):
        super(TinyVGG, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * hidden_units, out_shape),
        )
    def forward(self, x):
        x = self.conv_block_1(x)
        # print("Conv block 1:", x.shape)
        x = self.conv_block_2(x)
        # print("Conv block 2:", x.shape)
        x = self.classifier(x)
        # print("Classifier:", x.shape)
        return x

# from torchinfo import summary

# demo_image = simple_train_data_loader.dataset[0]
# print("Demo Image:", demo_image)

# model = TinyVGG(in_shape=3, hidden_units=32, out_shape=len(class_names))
# print("TinyVGG:", model)
# print("TinyVGG summary:", summary(model, input_size=(1, 3, 64, 64), verbose=0))

# demo_image_output = model(demo_image[0].unsqueeze(0))
# print("TinyVGG output:", demo_image_output)

# demo_image_softmax = F.softmax(demo_image_output, dim=1)
# print("TinyVGG softmax output:", demo_image_softmax)

# demo_image_pred = torch.argmax(demo_image_softmax, dim=1)
# print("TinyVGG prediction:", demo_image_pred)

import tqdm

# Train step
def train_step(model, dataloader, loss_fn, optimizer, device="cpu"):
    # set model to train mode
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for _, (inputs, targets) in enumerate(tqdm.tqdm(dataloader)):
        # move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        # forward pass
        outputs = model(inputs)
        # calculate loss
        loss = loss_fn(outputs, targets)
        # zero the parameter gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update parameters
        optimizer.step()
        # calculate loss and accuracy
        train_loss += loss.item()
        train_acc += (outputs.argmax(dim=1) == targets).float().mean().item()

    return train_loss / len(dataloader), train_acc / len(dataloader)

# Test step
def test_step(model, dataloader, loss_fn, device="cpu"):
    # set model to eval mode
    model.eval()

    test_loss = 0.0
    test_acc = 0.0

    with torch.inference_mode():
        for _, (inputs, targets) in enumerate(tqdm.tqdm(dataloader)):
            # move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass
            outputs = model(inputs)
            # calculate loss
            loss = loss_fn(outputs, targets)
            # calculate loss and accuracy
            test_loss += loss.item()
            test_acc += (outputs.argmax(dim=1) == targets).float().mean().item()

    return test_loss / len(dataloader), test_acc / len(dataloader)

# Training
# seed 42
torch.manual_seed(42)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# model
model = TinyVGG(in_shape=3, hidden_units=32, out_shape=len(class_names))
model.to(device)

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs
EPOCHS = 7

# history
history = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": [],
}

import time

def train(model, train_data_loader, test_data_loader, loss_fn, optimizer, device, epochs):
    start = time.time()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        train_loss, train_acc = train_step(model, train_data_loader, loss_fn, optimizer, device)
        print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        test_loss, test_acc = test_step(model, test_data_loader, loss_fn, device)
        print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
        print()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")

train(model, simple_train_data_loader, simple_test_data_loader, loss_fn, optimizer, device, EPOCHS)

history_df = pd.DataFrame(history, columns=["train_loss", "train_acc", "test_loss", "test_acc"])
print("History:")
print(history_df)

# Plot train and test history in two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
history_df[["train_loss", "test_loss"]].plot(ax=ax1)
ax1.set(title="Loss", xlabel="Epoch", ylabel="Loss")
history_df[["train_acc", "test_acc"]].plot(ax=ax2)
ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
plt.show()
