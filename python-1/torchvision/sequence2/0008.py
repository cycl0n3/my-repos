import torch
from torch import nn

import matplotlib.pyplot as plt

# seed 42
torch.manual_seed(42)

weight = 0.7
bias = 0.3

start = 0
stop = 1
step = 0.01

# create a list of x values
x = torch.arange(start, stop, step).unsqueeze(1)
y = weight * x + bias

training = int(0.8 * len(x))
validation = len(x) - training

# split the data
X_train, X_val, y_train, y_val = x[:training], x[training:], y[:training], y[training:]

# print some values
print(f"X_train:\n{X_train[:5]}")
print(f"\ny_train:\n{y_train[:5]}")
print(f"\nX_val:\n{X_val[:5]}")
print(f"\ny_val:\n{y_val[:5]}")


class CircleModelV1(nn.Module):
    def __init__(self):
        super(CircleModelV1, self).__init__()
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, x):
        z = self.layer1(x)
        z = self.layer2(z)
        z = self.layer3(z)
        return z


# define the model
model_1 = CircleModelV1()

# define the loss function
loss_function = nn.L1Loss()

# define the optimizer
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01)

# define the number of epochs
epochs = 5000

# train the model
for epoch in range(epochs):
    # set the model to train mode
    model_1.train()

    # forward pass
    y_pred = model_1(X_train)

    # calculate the loss
    loss = loss_function(y_pred, y_train)

    # zero the gradients
    optimizer.zero_grad()

    # backward pass
    loss.backward()

    # update the weights
    optimizer.step()

    # print the loss every 100 epochs
    if (epoch + 1) % 1000 == 0:
        with torch.inference_mode():
            # set the model to evaluation mode
            model_1.eval()

            # make predictions
            y_val_pred = model_1(X_val)

            # calculate the validation loss
            val_loss = loss_function(y_val_pred, y_val)

            # print the loss
            print(f"Epoch {epoch} | Training loss: {loss.item()} | Test loss: {val_loss.item()}")


# test the model
with torch.inference_mode():
    # set the model to evaluation mode
    model_1.eval()

    # make predictions
    y_pred = model_1(X_val)

    # calculate the loss
    loss = loss_function(y_pred, y_val)

    # print the loss
    print(f"Test loss: {loss.item()}")

    # plot the training and validation data
    plt.plot(X_train, y_train, "b.", label="Train")
    plt.plot(X_val, y_val, "r.", label="Validation")
    plt.plot(X_val, y_pred.detach().numpy(), "g.", label="Predictions")
    plt.legend()
    plt.show()

