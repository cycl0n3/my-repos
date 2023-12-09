import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn.functional as F

import torchvision.transforms as transforms

import os
import tqdm
import random

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# data transforms
transformer = transforms.Compose([
    
    transforms.Resize((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.Grayscale(num_output_channels=3),
    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    # transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
    transforms.ToTensor(),
])

# Load data
data = datasets.Caltech101(
    root="data",
    target_type="category",
    download=True,
    transform=transformer
)

print(f"Data: {data}")

# Split data into training and test sets
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

training_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

classes = []
CLASSES_PATH = "data\\caltech101\\101_ObjectCategories"

for d in os.listdir(CLASSES_PATH):
    if d != "BACKGROUND_Google":
        classes.append(d)

print(f"Training samples: {len(training_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Classes: {len(classes)}")

# Create data loaders
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Create the model
class NeuralNetwork(nn.Module):
    def __init__(self, in_shape=3, hidden_shape=16, out_shape=101):
        super(NeuralNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_shape, hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_shape, hidden_shape, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*hidden_shape, out_shape),
        )

    def forward(self, x):
        x = self.layer1(x)#; print("Layer 1:", x.shape)
        x = self.layer2(x)#; print("Layer 2:", x.shape)
        x = self.layer3(x)#; print("Layer 3:", x.shape)
        return x


model = NeuralNetwork().to(device)
print(model)

# Optimizer and loss function
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# sample image from the dataset
sample_image, sample_label = training_data[0]
sample_image = sample_image.unsqueeze(0).to(device)

print("Sample image shape:", sample_image.shape)
print("Sample label:", sample_label)

# sample image prediction
sample_pred = model(sample_image)
print("Sample prediction:", sample_pred)

# sample image softmax
sample_pred_softmax = F.softmax(sample_pred, dim=1)
print("Sample prediction softmax:", sample_pred_softmax)

# sample image prediction label
sample_pred_label = torch.argmax(sample_pred_softmax, dim=1)
print("Sample prediction label:", classes[sample_pred_label])

def test_plot(dataloader, model):
    # select 9 random images
    images, labels = next(iter(dataloader))
    images = images.to(device)[:9]
    labels = labels.to(device)[:9]

    # get prediction
    model.eval()
    preds = model(images)
    preds_softmax = F.softmax(preds, dim=1)
    preds_label = torch.argmax(preds_softmax, dim=1)
    preds_label = [classes[i] for i in preds_label]
    actual_labels = [classes[i] for i in labels]

    # plot images
    fig = plt.figure(figsize=(12, 12))
    for i in range(0, 9):
        ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
        ax.imshow(images[i].squeeze().permute(1, 2, 0))
        ax.set_title(f"Actual: {actual_labels[i]} \n Prediction: {preds_label[i]}")
    plt.show()


test_plot(train_dataloader, model)

# plot sample image
# title = f"Label: {classes[sample_label]} | Prediction: {classes[sample_pred_label]}"
# plt.imshow(sample_image.squeeze().permute(1, 2, 0))
# plt.title(title)
# plt.show()

# x = 1 / 0

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    train_loss, train_accuracy = 0, 0

    for _, (X, y) in enumerate(tqdm.tqdm(dataloader)):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_loss += loss.item()

        # Compute accuracy
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        train_accuracy += correct

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= size
    train_accuracy /= size

    print(f"Train Error: \n Accuracy: {(100*train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f} \n")

# Test loop
def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, test_accuracy = 0, 0

    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    test_accuracy /= size

    print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# MODEL_PATH = "models\\0023.pth"

# if os.path.exists(MODEL_PATH):
#     model.load_state_dict(torch.load(MODEL_PATH))
#     print("Loaded model from disk")


# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model)
# print("Done!")

# torch.save(model.state_dict(), MODEL_PATH)
# print("Saved model to disk")
