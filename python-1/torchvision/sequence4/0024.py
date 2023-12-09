import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torchinfo

import torch.nn.functional as F

import torchvision.transforms as transforms

import os
import tqdm
import random

import pandas as pd

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

WIDTH = 128
HEIGHT = 128

# data transforms
transformer = transforms.Compose([
    # Resize the image
    transforms.Resize((WIDTH, HEIGHT)),

    # Data augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    
    # Convert the image to grayscale
    transforms.Grayscale(num_output_channels=3),
    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    # transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
    transforms.ToTensor(),
])

DATA_PATH = "data\\caltech101\\101_ObjectCategories"

# Load data
data = datasets.ImageFolder(DATA_PATH, transform=transformer)

# Load data classes
classes = data.classes
classes_to_index = data.class_to_idx
index_to_classes = {v: k for k, v in classes_to_index.items()}

print(f"Classes: {classes}")
print(f"Classes: {len(classes)}")
print(f"Classes to index: {classes_to_index}")
print(f"Index to classes: {index_to_classes}")

# Split data into training and test sets
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

training_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

print(f"Training samples: {len(training_data)}")
print(f"Test samples: {len(test_data)}")

# Create data loaders
BATCH_SIZE = 64

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Create model
class NeuralNet24(nn.Module):
    def __init__(self, in_shape, hidden_units, out_shape):
        super(NeuralNet24, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_shape, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 32 * 32 * hidden_units, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)
        # print("Layer 1:", x.shape)
        x = self.layer2(x)
        # print("Layer 2:", x.shape)
        return x


# Initialize model
model = NeuralNet24(in_shape=3, hidden_units=32, out_shape=len(classes))

# Sample image
sample = torch.rand(1, 3, WIDTH, HEIGHT)

# Model summary
summary = torchinfo.summary(model, input_data=sample)
print(summary)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_accuracy, train_loss = 0, 0

    model.train()
    for _, (X, y) in enumerate(tqdm.tqdm(dataloader)):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()

        # Accuracy
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        train_accuracy += correct

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= size
    train_accuracy /= size

    return train_loss, train_accuracy

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_accuracy, test_loss = 0, 0

    model.eval()
    with torch.no_grad():
        for _, (X, y) in enumerate(tqdm.tqdm(dataloader)):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()

            # Accuracy
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            test_accuracy += correct

    test_loss /= size
    test_accuracy /= size

    return test_loss, test_accuracy

history = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
}

EPOCHS = 5

MODEL_PATH = "models\\0024.pth"

if os.path.exists(MODEL_PATH):
    print("Loading model...")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded!")

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_accuracy = train(train_dataloader, model, criterion, optimizer)
    test_loss, test_accuracy = test(test_dataloader, model, criterion)
    print(f"Train: Accuracy: {(100*train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f}")
    print(f"Test: Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["test_loss"].append(test_loss)
    history["test_accuracy"].append(test_accuracy)
    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved!")

print("Done!")

accuracy_and_loss = pd.DataFrame(history)
accuracy_and_loss.plot()
plt.show()
