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
model3 = FashionMNISTModelV3(input_features=1, hidden_units=32, output_features=len(classes))
learning_rate = 0.1

model = model3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# test_conv = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)

# test_image = torch.rand(1, 3, 64, 64)
# print("Test image shape:", test_image.shape)

# test_output = test_conv(test_image)
# print("Test image after conv shape:", test_output.shape)

# max_pool_layer = nn.MaxPool2d(kernel_size=2)

# test_max_pool = max_pool_layer(test_output)
# print("Test image after max pool shape:", test_max_pool.shape)

# div()

# random_tensor = torch.rand(1, 1, 2, 2)
# print("Random tensor:", random_tensor)
# print("Random tensor shape:", random_tensor.shape)

# max_pool_layer = nn.MaxPool2d(kernel_size=2)

# max_pool_tensor = max_pool_layer(random_tensor)
# print("Max pool tensor:", max_pool_tensor)
# print("Max pool tensor shape:", max_pool_tensor.shape)

# div()

# image0 = fashion_mnist_train_data[0][0]
# print("Image shape:", image0.shape)

# image0_out = model(image0.unsqueeze(0))
# print("Image out:", image0_out)
# print("Image out shape:", image0_out.shape)

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
        
        for images, targets in tqdm.tqdm(test_data_loader):
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

MODEL_VERSION = "v3_a"
MODEL_PATH = "models/fashion_mnist_model_" + MODEL_VERSION + ".pth"

epochs = 2

if not os.path.exists(MODEL_PATH):
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        # Training
        train_loss, train_accuracy = train(model, criterion, optimizer, train_data_loader)
        print("Train loss:", train_loss)
        print("Train accuracy:", train_accuracy)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        # Testing
        test_loss, test_accuracy = test(model, criterion, test_data_loader)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)
        # save the model
        torch.save(model.state_dict(), MODEL_PATH)

        # Test the model
        model.load_state_dict(torch.load(MODEL_PATH))
        test_loss, test_accuracy = test(model, criterion, test_data_loader)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_accuracy)
else:
    # Load the model
    model.load_state_dict(torch.load(MODEL_PATH))

# Plot the loss and accuracy
# plt.figure(figsize=(10, 5))
# plt.title("Loss")
# plt.plot(train_loss_history, label="train")
# plt.plot(test_loss_history, label="test")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

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


test_samples = []
test_labels = []

# seed 42
np.random.seed(42)
torch.manual_seed(42)

# random 9 items in fashion_mnist_test_data
for image, label in random.sample(list(fashion_mnist_test_data), 9):
    test_samples.append(image)
    test_labels.append(label)

# view first sample shape
print("Test sample shape:", test_samples[0].shape)

# make predictions
probabilities = make_predictions(model, test_samples)
# convert to tensor
probabilities = torch.cat(probabilities)
print("Labels:", test_labels)
print("Probabilities:", probabilities.argmax(dim=1))

# # plot the samples
# plt.figure(figsize=(10, 10))
# for i in range(len(test_samples)):
#     plt.subplot(3, 3, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     actual_label = test_labels[i]
#     predicted_label = probabilities[i].argmax(dim=0)
#     title_text = classes[actual_label] + "::" + classes[predicted_label]
#     if actual_label == predicted_label:
#         plt.title(title_text, color="green")
#     else:
#         plt.title(title_text, color="red")
#     plt.imshow(test_samples[i][0], cmap=plt.cm.binary)
# plt.show()
