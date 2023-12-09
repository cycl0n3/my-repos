import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import os
import tqdm

import random
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    train_loss, correct = 0, 0
    
    for _, (X, y) in enumerate(tqdm.tqdm(dataloader)):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    correct /= size
    print(f"Train: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def start():
    epochs = 3
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

model = NeuralNetwork()
MODEL_VERSION = 8
MODEL_NN_PATH = f"models\\model_nn_v{MODEL_VERSION}.txt"
MODEL_PATH = f"models\\model_v{MODEL_VERSION}.pth"

# seed = 42
torch.manual_seed(42)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Loaded PyTorch Model State from {MODEL_PATH}")
    test(test_dataloader, model, loss_fn)
else:
    start()
    torch.save(model.state_dict(), MODEL_PATH)
    with open(MODEL_NN_PATH, "w") as f:
        f.write(str(model))
    print(f"Saved PyTorch Model State to {MODEL_PATH}")

classes = train_dataloader.dataset.classes

model.eval()

# select 9 random images from test dataset
indices = random.sample(range(len(test_data)), 9)
images = torch.stack([test_data[i][0] for i in indices])
labels = [test_data[i][1] for i in indices]

with torch.no_grad():
    pred = model(images)
    actual_labels = [classes[labels[i]] for i in range(len(labels))]
    predicted_classes = pred.argmax(1)

predicted_labels = [classes[predicted_classes[i]] for i in range(len(predicted_classes))]

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    ax.imshow(images[i].squeeze(), cmap="gray")
    color = 'green' if predicted_labels[i] == actual_labels[i] else 'red'
    title = f"{predicted_labels[i]} ({actual_labels[i]})"
    ax.set_title(title, fontsize=10, color=color)
    ax.axis("off")
plt.show()
