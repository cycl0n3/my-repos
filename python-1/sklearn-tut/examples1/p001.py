import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Load the iris dataset
iris = datasets.load_iris()
print("Description: ", iris.DESCR)

# Print the iris features
features = iris.feature_names
print("Features: ", features)

# Print the iris labels
targets = iris.target_names
print("Labels: ", targets)

# First 5 rows of data
print("First 5 rows of data: ", iris.data[:5])

item0 = iris.data[0]
print("First item: ", item0)

# plot the sepal length vs sepal width
x_index = 0
y_index = 1

# this formatter will label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

plt.figure(figsize=(5, 4))
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

plt.tight_layout()
plt.show()
