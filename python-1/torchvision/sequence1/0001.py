import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.7
bias = 0.3

X = torch.arange(0, 1, 0.02).unsqueeze(1)
Y = weight * X + bias

train_split_x = X[:int(len(X) * 0.8)]
train_split_y = Y[:int(len(Y) * 0.8)]

test_split_x = X[int(len(X) * 0.8):]
test_split_y = Y[int(len(Y) * 0.8):]

predictions = None

# plot the data
plt.plot(train_split_x, train_split_y, 'b.')
plt.plot(test_split_x, test_split_y, 'g.')
if predictions is not None:
    plt.plot(test_split_x, predictions, 'r.')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['train', 'test', 'predictions'])
plt.show()
