import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

from pathlib import Path

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using {} device".format(device))

MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

MODEL_NAME = "project1-L1.pt"
MODEL_SAVE_PATH = MODELS / MODEL_NAME


# save the model state dict
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x):
        return self.weights * x + self.bias


seed = 42
torch.manual_seed(seed)

model_0 = LinearRegressionModel()

if MODEL_SAVE_PATH.exists():
    model_0 = load_model(model_0, MODEL_SAVE_PATH)
    print("Existing model loaded")
else:
    print("New model created")

print(model_0.state_dict())

weight, bias = 0.7, 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(1)
Y = weight * X + bias
x_train = X[:int(len(X) * 0.8)]
y_train = Y[:int(len(Y) * 0.8)]
x_test = X[int(len(X) * 0.8):]
y_test = Y[int(len(Y) * 0.8):]

# mean square error loss function
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []


def train_loop():
    for epoch in range(epochs):
        # Set model to training mode
        model_0.train()
        # Forward pass
        predictions_y = model_0(x_train)
        # Compute Loss
        loss = loss_fn(predictions_y, y_train)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Reset gradients to zero
        optimizer.zero_grad()
        # Set model to evaluation mode
        model_0.eval()

        with torch.inference_mode():
            test_predictions_y = model_0(x_test)
            test_predictions_y_loss = loss_fn(test_predictions_y, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss.item())
            test_loss_values.append(test_predictions_y_loss.item())
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_predictions_y_loss.item():.4f}')

    save_model(model_0, MODEL_SAVE_PATH)


if not MODEL_SAVE_PATH.exists():
    train_loop()
else:
    print("Model already trained")


model_0.eval()

# Plot the training data against the model prediction
with torch.inference_mode():
    predictions_y = model_0(x_train)
    test_predictions_y = model_0(x_test)

plt.plot(x_train, y_train, 'r.', label='Training data')
plt.plot(x_train, predictions_y.detach().numpy(), label='Prediction')
plt.plot(x_test, y_test, 'b.', label='Test data')
plt.plot(x_test, test_predictions_y.detach().numpy(), label='Test Prediction')
plt.legend()
plt.show()
