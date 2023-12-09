import torch
from torch import nn
import matplotlib.pyplot as plt


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
print(model_0.state_dict())

weight, bias = 0.7, 0.3
X = torch.arange(0, 1, 0.02).unsqueeze(1)
Y = weight * X + bias
x_train = X[:int(len(X) * 0.8)]
y_train = Y[:int(len(Y) * 0.8)]
x_test = X[int(len(X) * 0.8):]
y_test = Y[int(len(Y) * 0.8):]

# with torch.inference_mode():
#     predictions_y = model_0(x_test)
#     print(predictions_y)


# loss function
loss_fn = nn.L1Loss()

# Optimizer
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01)

# Training loop
epochs = 200

epoch_count = []
loss_values = []
test_loss_values = []

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


# Print model parameters
print(model_0.state_dict())

predictions = model_0(x_test).detach().numpy()

# Calculate accuracy
accuracy = ((predictions - y_test.detach().numpy()) / y_test).abs().mean()
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save model to file
torch.save(model_0.state_dict(), 'model_0.pth')

# Plot the loss values
plt.plot(epoch_count, loss_values, 'r--')
plt.plot(epoch_count, test_loss_values, 'b-')
plt.legend(['Training Loss', 'Test Loss'])

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
