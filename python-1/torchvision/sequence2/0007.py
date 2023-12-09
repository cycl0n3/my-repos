import torch
from torch import nn

import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import pandas as pd

from helpers import helper_functions as hf

n_samples = 1000

X, y = make_circles(
    n_samples=n_samples,
    noise=0.03,
    random_state=42,
    shuffle=True,
)

# print(f"First 5 X features:\n{X[:5]}")
# print(f"\nFirst 5 y labels:\n{y[:5]}")

circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
# print(circles.head(10))

# Turn data into tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# view first 5 rows of X, y
# print(X[:5], y[:5])

# split the data into training and validation sets
train_size = int(0.8 * len(X))
val_size = len(X) - train_size

# split the data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)


# define the model
class CircleModelV0(nn.Module):
    def __init__(self):
        super(CircleModelV0, self).__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class CircleModelV1(nn.Module):
    def __init__(self):
        super(CircleModelV1, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        z = self.layer1(x)
        z = self.layer2(z)
        z = self.layer3(z)
        return z


# initialize the model
# model = CircleModelV0()
# print(model.state_dict())

# model
model = CircleModelV1()
print(model.state_dict())

with torch.inference_mode():
    y_pred = model(X_train)
    print(y_pred[:10])

# loss function
loss_fn = nn.BCEWithLogitsLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# training loop
epochs = 1000


# accuracy function
def accuracy_fn(y_preds, y_true):
    y_preds = torch.round(torch.sigmoid(y_preds))
    correct = (y_preds == y_true).float()
    acc = correct.sum() / len(correct)
    return acc * 100


# seed 42
torch.manual_seed(42)


def train():
    for epoch in range(epochs):
        # set model to train mode
        model.train()

        # forward pass
        y_logits = model(X_train).squeeze(1)

        # prediction
        y_preds = torch.round(torch.sigmoid(y_logits))

        # calculate loss
        loss = loss_fn(y_logits, y_train)

        # calculate accuracy
        acc = accuracy_fn(y_preds, y_train)

        # backward pass
        loss.backward()

        # update parameters
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        # Test
        model.eval()

        with torch.inference_mode():
            test_y_logits = model(X_val).squeeze(1)
            test_y_preds = torch.round(torch.sigmoid(test_y_logits))
            test_y_loss = loss_fn(test_y_logits, y_val)
            test_y_acc = accuracy_fn(test_y_preds, y_val)

        # print loss
        if (epoch + 1) % 200 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Acc(%): {acc.item():.2f}%", end=" ")
            print(f"Test Loss: {test_y_loss.item():.4f}, Test Acc(%): {test_y_acc.item():.2f}%")


# train the model
train()

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
hf.plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
hf.plot_decision_boundary(model, X_val, y_val)
plt.show()
