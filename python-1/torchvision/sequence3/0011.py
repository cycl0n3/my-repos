from timeit import default_timer as timer

import torch
import torch.nn as nn
import torchvision
import torchmetrics

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import tqdm

# seed 42
np.random.seed(42)
torch.manual_seed(42)

# print version
print("Pytorch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)

# Get the data
fashion_mnist_train_data = datasets.FashionMNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
fashion_mnist_test_data = datasets.FashionMNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)
print("Train data:", fashion_mnist_train_data)
print("Test data:", fashion_mnist_test_data)
print("Train data size:", len(fashion_mnist_train_data))
print("Test data size:", len(fashion_mnist_test_data))

classes_idx = fashion_mnist_train_data.class_to_idx
classes = fashion_mnist_train_data.classes
print("Classes Index:", classes_idx)
print("Classes Array:", classes)

# Create data loaders
batch_size = 64
train_data_loader = torch.utils.data.DataLoader(fashion_mnist_train_data, batch_size=batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(fashion_mnist_test_data, batch_size=batch_size, shuffle=False)

# Get the first batch
for images, labels in train_data_loader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    break

# Create the model v1
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super(FashionMNISTModelV1, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden_units),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        return self.layer_stack(x)


# Create the model v2
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super(FashionMNISTModelV2, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer_stack(x)


# Create the model v3 using convolutional layers
class FashionMNISTModelV3(nn.Module):
    def __init__(self, input_features: int, hidden_units: int, output_features: int):
        super(FashionMNISTModelV3, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_features, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 7 * 7, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        print(x.shape)
        x = self.conv_block_2(x)
        print(x.shape)
        return self.classifier(x)

# Training
#model1 = FashionMNISTModelV0(input_features=28 * 28, hidden_units=128, output_features=len(classes))
#model2 = FashionMNISTModelV2(input_features=28 * 28, hidden_units=128, output_features=len(classes))
model3 = FashionMNISTModelV3(input_features=1, hidden_units=32, output_features=len(classes))
learning_rate = 0.01

model = model3

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def print_time(label: str, start: float, end: float):
    print(f"{label} time: {end - start:.2f}s")

epochs = 3

def train(loader, model):
    train_accuracy = 0.0
    train_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loader):
        # Set model to train mode
        model.train()
        # Get data to cuda if possible
        if torch.cuda.is_available():
            data = data.cuda()
            targets = targets.cuda()

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()

        # accuracy
        accuracy = torchmetrics.functional.accuracy(scores, targets, task="multiclass", num_classes=len(classes))
        train_accuracy += accuracy.item()

        # gradient descent or adam step
        optimizer.step()

    train_accuracy /= len(loader)
    train_loss /= len(loader)

    print(f"Train Loss: {train_loss:.2f}%, Train Accuracy: {train_accuracy*100:.2f}%")


def test(loader, model):
    print("Testing")

    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    # num_correct = 0
    # num_samples = 0

    model.eval()

    with torch.inference_mode():
        loss = 0.0
        accuracy = 0.0

        for x, y in tqdm.tqdm(loader):
            # Get data to cuda if possible
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # forward
            scores = model(x)
            loss += criterion(scores, y).item()
            accuracy += torchmetrics.functional.accuracy(scores, y, task="multiclass", num_classes=len(classes))
            # _, predictions = scores.max(1)
            # num_correct += (predictions == y).sum()
            # num_samples += predictions.size(0)

        #print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}")
        print(f"Loss: {loss / len(loader)}, Accuracy: {accuracy / len(loader)}")

    model.train()

# Train and check accuracy
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    start = timer()
    train(train_data_loader, model)
    end = timer()
    print_time("Train", start, end)
    start = timer()
    test(test_data_loader, model)
    end = timer()
    print_time("Test", start, end)
    print()
