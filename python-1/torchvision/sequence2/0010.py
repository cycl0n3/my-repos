import torch
import torch.nn as nn

import torchaudio
print(torchaudio.__version__)
print(str(torchaudio.get_audio_backend()))

from torchmetrics import Accuracy

import pandas as pd

import seaborn as sns
sns.set_theme(style="darkgrid")

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# seed 42
torch.manual_seed(42)

# sample size
n_samples = 1000

# no of classes
n_classes = 4

# no of features
n_features = 2

# generate data
X, y = make_blobs(
    n_samples,
    centers=n_classes,
    cluster_std=1.5,
    n_features=n_features,
    random_state=42,
)

# turn data into tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()

# train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)

# view first 10 rows of X, y tensors as pandas dataframes
data_frame = pd.DataFrame(X.numpy())
data_frame.columns = ["X0", "X1"]
data_frame["classification"] = [chr(int(label) + ord('A')) for label in y.numpy()]
print(data_frame.head(5))

# define the model
model = nn.Sequential(
    nn.Linear(n_features, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, n_classes),
)

# define the loss function
loss_function = nn.CrossEntropyLoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# accuracy function
def accuracy(y_preds, y_true):
    return torch.sum(y_preds == y_true).float() / len(y_true)


# accuracy metric
acc_metric = Accuracy(task="multiclass", num_classes=n_classes)

# define the training loop
n_epochs = 200

for epoch in range(n_epochs):
    model.train()
    # forward pass
    y_logits = model(X_train)
    y_pred_probs = torch.softmax(y_logits, dim=1)
    y_preds = torch.argmax(y_pred_probs, dim=1)
    loss = loss_function(y_logits, y_train.long())
    #acc = accuracy(y_preds, y_train) * 100
    acc = acc_metric(y_preds, y_train.long()) * 100

    # backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # print loss
    if epoch % 10 == 0:
        with torch.inference_mode():
            test_y_logits = model(X_test)
            test_y_pred_probs = torch.softmax(test_y_logits, dim=1)
            test_y_preds = torch.argmax(test_y_pred_probs, dim=1)
            test_loss = loss_function(test_y_logits, y_test.long())
            # test_acc = accuracy(test_y_preds, y_test) * 100
            test_acc = acc_metric(test_y_preds, y_test.long()) * 100
        print(f"Epoch: {epoch} | Train Loss: {loss:.2f} | Train Acc: {acc:.2f}% | Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%")


test_data = X_test[0:10]
test_label = y_test[0:10]
print(f"Test Data: {test_data}")
print(f"Test Label: {test_label}")

# predict
with torch.inference_mode():
    test_data_logits = model(test_data)
    test_data_pred_probs = torch.softmax(test_data_logits, dim=1)
    test_data_preds = torch.argmax(test_data_pred_probs, dim=1)
    print(f"Test Data Prediction: {test_data_preds}")


# calculate decision boundary
x0_min, x0_max = data_frame["X0"].min() - 1, data_frame["X0"].max() + 1
x1_min, x1_max = data_frame["X1"].min() - 1, data_frame["X1"].max() + 1
x0_mesh, x1_mesh = torch.meshgrid(torch.arange(x0_min, x0_max, 0.1), torch.arange(x1_min, x1_max, 0.1), indexing="ij")
x0_mesh_flat = x0_mesh.flatten()
x1_mesh_flat = x1_mesh.flatten()
X_mesh = torch.stack((x0_mesh_flat, x1_mesh_flat), dim=1)
y_mesh = model(X_mesh)
y_mesh = torch.argmax(y_mesh, dim=1)
y_mesh = y_mesh.reshape(x0_mesh.shape)

# plot the decision boundary
plt.contourf(x0_mesh, x1_mesh, y_mesh, cmap=plt.cm.Spectral, alpha=0.3)

# plot the data
sns.scatterplot(data=data_frame, x="X0", y="X1", hue="classification")
plt.show()
