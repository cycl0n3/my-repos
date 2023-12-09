import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from pathlib import Path

# create data using the linear regression formula y = weight * x + bias
weight, bias = 0.7, 0.3

# create range values
start = 0
end = 1
step = 0.03

# seed 42
torch.manual_seed(42)

# create x and y values
x = torch.arange(start, end, step).unsqueeze(1)
y = weight * x + bias

# split the data into training and validation sets
train_size = int(0.8 * len(x))
val_size = len(x) - train_size

# split the data
x_train, x_val = torch.split(x, [train_size, val_size])
y_train, y_val = torch.split(y, [train_size, val_size])

print(f"Training data shape: {x_train.shape}")
print(f"Validation data shape: {x_val.shape}")


# plot prediction
def plot_predictions(xtrain, ytrain, xtest, ytest, predictions=None):
    fig, ax = plt.subplots()
    ax.plot(xtrain, ytrain, "b-", label="Training data")
    ax.plot(xtest, ytest, "r-", label="Validation data")
    if predictions is not None:
        ax.plot(xtest, predictions, "g.", label="Predictions")
    ax.legend()
    plt.show()


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super(LinearRegressionModelV2, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


MODEL_SAVE_PATH = Path("models/linear_regression_model.pt")

model0 = LinearRegressionModelV2()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model0.parameters(), lr=1e-2)
epochs = 200


# Training loop
def train():
    for epoch in range(epochs):
        # Set model to training mode
        model0.train()
        # Forward pass
        predictions_y = model0(x_train)
        # Compute Loss
        loss = loss_fn(predictions_y, y_train)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Reset the gradients to zero
        optimizer.zero_grad()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
            # Calculate test loss
            with torch.inference_mode():
                test_predictions_y = model0(x_val)
                test_predictions_y_loss = loss_fn(test_predictions_y, y_val)
                print(f"Test loss: {test_predictions_y_loss.item():.4f}")


# Save the model
if not MODEL_SAVE_PATH.parent.exists():
    train()
    torch.save(model0.state_dict(), MODEL_SAVE_PATH)
    print("Model saved")
    print(model0.state_dict())
else:
    print("Model already exists")
    model0.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(model0.state_dict())

# Plot the training data and the predictions of the untrained model
model0.eval()
with torch.inference_mode():
    test_predictions_y = model0(x_val)
    plot_predictions(x_train, y_train, x_val, y_val, test_predictions_y.detach())
