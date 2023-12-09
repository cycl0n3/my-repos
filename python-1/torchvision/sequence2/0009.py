import numpy as np
import pandas as pd

import seaborn as sns
sns.set_theme(style="darkgrid")

import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

import torch
from torch import nn

# seed 42
torch.manual_seed(42)

# sample size
n_samples = 1000

# generate data
X, y = make_circles(
    n_samples,
    noise=0.03,
    random_state=42,
)

# turn data into tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)

# view first 5 rows of X, y tensors as pandas dataframes
data_frame = pd.DataFrame(X.numpy())
data_frame.columns = ["X0", "X1"]
data_frame["class"] = y.numpy()
print(data_frame.head(5))


class CircleModelV2(nn.Module):
    def __init__(self):
        super(CircleModelV2, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# instantiate the model
model = CircleModelV2()
print(model.state_dict())

# define the loss function
criterion = nn.BCEWithLogitsLoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.12)


# accuracy function
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    # no. of correct predictions
    correct = (rounded_preds == y).float()
    # accuracy
    acc = correct.sum() / len(correct)

    return acc


# define the number of epochs
epochs = 1000

# training loop
for epoch in range(epochs):
    # set the model to train mode
    model.train()
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward pass
    output_logits = model(X_train).squeeze(1)
    # apply sigmoid to outputs
    output_predictions = torch.round(torch.sigmoid(output_logits))
    # calculate the loss
    loss = criterion(output_logits, y_train)
    # backpropagation
    loss.backward()
    # update the parameters
    optimizer.step()

    # print statistics
    if epoch % 100 == 0:
        with torch.inference_mode():
            # set the model to eval mode
            model.eval()
            # calculate the accuracy
            test_logits = model(X_test).squeeze(1)
            # apply sigmoid to outputs
            test_predictions = torch.round(torch.sigmoid(output_logits))
            # calculate loss
            test_loss = criterion(test_logits, y_test)
            # calculate the accuracy
            test_accuracy = binary_accuracy(test_logits, y_test)

            print(f"Epoch: {epoch} | Loss: {test_loss.item():.4f} | Accuracy: {test_accuracy:.4f}")


print(model.state_dict())

with torch.inference_mode():
    # set the model to eval mode
    model.eval()
    # calculate the accuracy
    output_logits = model(X_test).squeeze(1)
    # apply sigmoid to outputs
    output_predictions = torch.round(torch.sigmoid(output_logits))
    # calculate the accuracy
    accuracy = binary_accuracy(output_predictions, y_test)

    df = pd.DataFrame(X_test.numpy())
    df.columns = ["X0", "X1"]
    df["output_predictions"] = output_predictions.numpy()
    df["y_test"] = y_test.numpy()

    print(df.head(5))


# # plot the graph
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.title("Train")
# hf.plot_decision_boundary(model, X_train, y_train)
#
# plt.subplot(1, 2, 2)
# plt.title("Test")
# hf.plot_decision_boundary(model, X_test, y_test)
#
# plt.title("Circles")
# plt.xlabel("X0")
# plt.ylabel("X1")
#
# # show the plot
# plt.show()


# plot the graph
sns.scatterplot(data=data_frame, x="X0", y="X1", hue="class", palette="deep")

# calculate decision boundary
x_min, x_max = data_frame["X0"].min() - 0.1, data_frame["X0"].max() + 0.1
y_min, y_max = data_frame["X1"].min() - 0.1, data_frame["X1"].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# plot decision boundary
Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
Z = torch.round(torch.sigmoid(Z)).detach().numpy()
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)


plt.title("Circles")
plt.xlabel("X0")
plt.ylabel("X1")

# show the plot
plt.show()
