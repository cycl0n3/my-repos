from timeit import default_timer as timer

import os
import sys
import time
import random

import torch
import torch.nn as nn
import torchvision
import torchmetrics

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import mlxtend
import mlxtend.plotting as mlxplt

import tqdm

def div():
    print("---------------------------------------------------------------")

# seed 42
np.random.seed(42)
torch.manual_seed(42)

# print version
print("Pytorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
print("Torchmetrics Version:", torchmetrics.__version__)
print("MLXtend Version:", mlxtend.__version__)

# Get the data
fashion_mnist_train_data = datasets.FashionMNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
fashion_mnist_test_data = datasets.FashionMNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)

classes_idx = fashion_mnist_train_data.class_to_idx
classes = fashion_mnist_train_data.classes
# print("Classes Index:", classes_idx)
# print("Classes Array:", classes)

div()

# Create data loaders
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(fashion_mnist_train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(fashion_mnist_test_data, batch_size=batch_size, shuffle=False)
# print("Train data loader:", len(train_data_loader))
# print("Test data loader:", len(test_data_loader))

# subset of the train data at 25%
train_data_loader_subset_25 = torch.utils.data.DataLoader(fashion_mnist_train_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(list(range(0, len(fashion_mnist_train_data), 4))))
# print("Train data loader subset:", len(train_data_loader_subset_25))

# subset of the test data at 25%
test_data_loader_subset_25 = torch.utils.data.DataLoader(fashion_mnist_test_data, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(list(range(0, len(fashion_mnist_test_data), 4))))
# print("Test data loader subset:", len(test_data_loader_subset_25))

# x = 1 / 0

# Create the model v3 using convolutional layers
class FashionMNISTModelV3(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super(FashionMNISTModelV3, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_features, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, output_features),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        # print("Conv block 1:", x.shape)
        x = self.conv_block_2(x)
        # print("Conv block 2:", x.shape)
        x = self.classifier(x)
        # print("Classifier:", x.shape)
        return x

# Training
model3 = FashionMNISTModelV3(input_features=1, hidden_units=25, output_features=len(classes))
learning_rate = 0.1

model = model3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
def train(model, criterion, optimizer, train_data_loader):
    # Set model to training mode
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0

    # Training
    for images, targets in tqdm.tqdm(train_data_loader):
        # Zero out the optimizer
        optimizer.zero_grad()
        # Forward pass
        scores = model(images)
        # Loss calculation
        loss = criterion(scores, targets)
        train_loss += loss.item()
        # Accuracy calculation
        accuracy = torchmetrics.functional.accuracy(scores, targets, num_classes=len(classes), task="multiclass")
        train_accuracy += accuracy.item()
        # Backward pass
        loss.backward()
        # Optimizer step
        optimizer.step()
    
    # Average loss and accuracy
    train_loss /= len(train_data_loader)
    train_accuracy /= len(train_data_loader)

    return train_loss, train_accuracy

# Testing loop
def test(model, criterion, test_data_loader):
    test_loss = 0.0
    test_accuracy = 0.0

    # Testing
    with torch.inference_mode():
        # Set model to evaluation mode
        model.eval()
        
        for images, targets in tqdm.tqdm(test_data_loader, colour="red"):
            # Forward pass
            scores = model(images)
            # Loss calculation
            loss = criterion(scores, targets)
            test_loss += loss.item()
            # Accuracy calculation
            accuracy = torchmetrics.functional.accuracy(scores, targets, num_classes=len(classes), task="multiclass")
            test_accuracy += accuracy.item()
    
    # Average loss and accuracy
    test_loss /= len(test_data_loader)
    test_accuracy /= len(test_data_loader)

    return test_loss, test_accuracy

# Training and testing
train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

MODEL_VERSION = "v3_b"
MODEL_PATH = "models/fashion_mnist_model_" + MODEL_VERSION + ".pth"

epochs = 2

if not os.path.exists(MODEL_PATH):
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        # Training
        train_loss, train_accuracy = train(model, criterion, optimizer, train_data_loader)
        print("Train loss:", train_loss)
        print("Train accuracy:", train_accuracy)
        # save the model
        torch.save(model.state_dict(), MODEL_PATH)
else:
    # Load the model
    model.load_state_dict(torch.load(MODEL_PATH))
    # Test the model
    test_loss, test_accuracy = test(model, criterion, test_data_loader)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)

def make_predictions(model: nn.Module, images: list):
    probabilities = []

    # Testing
    with torch.inference_mode():
        # Set model to evaluation mode
        model.eval()
        
        # Forward pass
        for image in images:
            scores = model(image.unsqueeze(0))
            scores = torch.softmax(scores, dim=1)
            probabilities.append(scores)
    
    return probabilities


predictions = []
model.eval()

with torch.inference_mode():
    for images, targets in tqdm.tqdm(test_data_loader, colour="green"):
        scores = model(images)
        score_softmax = torch.softmax(scores, dim=1)
        score_argmax = torch.argmax(score_softmax, dim=1)
        predictions.append(score_argmax)

predictions = torch.cat(predictions)
print("Predictions:", predictions)
print("Predictions shape:", predictions.shape)

# Confusion matrix
confusion_matrix = torchmetrics.functional.confusion_matrix(
    predictions, 
    fashion_mnist_test_data.targets, 
    num_classes=len(classes), 
    # normalize="true",
    task="multiclass"
)
print("Confusion matrix:", confusion_matrix)
print("Confusion matrix shape:", confusion_matrix.shape)

# Plot the confusion matrix
fig, ax = mlxplt.plot_confusion_matrix(confusion_matrix.numpy(), class_names=classes)
plt.show()
